from __future__ import absolute_import, division, print_function

import os
import sys
import math
import time
import copy
import pickle
import socket
import random
import logging
import argparse
import numpy as np
from enum import Enum

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

import torchvision

# from models.ViT import VisionTransformer, CONFIGS
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mlp import MLP_sigmoid
from utils.data_utils import get_loader
from isda import ISDALoss
from meta import MetaSGD

logger = logging.getLogger(__name__)
best_acc1, best_epoch = 0.0, 0


""" some tools """
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, init_lr, epoch_total, warmup_epochs, epoch_cur, num_iter_per_epoch, i_iter):
    """
    cosine learning rate with warm-up
    """
    if epoch_cur < warmup_epochs:
        # T_cur = 1, 2, 3, ..., (T_total - 1)
        T_cur = 1 + epoch_cur * num_iter_per_epoch + i_iter
        T_total = 1 + warmup_epochs * num_iter_per_epoch
        lr = (T_cur / T_total) * init_lr
    else:
        # T_cur = 0, 1, 2, 3, ..., (T_total - 1)
        T_cur = (epoch_cur - warmup_epochs) * num_iter_per_epoch + i_iter
        T_total = (epoch_total - warmup_epochs) * num_iter_per_epoch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_meta_learning_rate(optimizer, init_lr, epoch_total, warmup_epochs, epoch_cur, num_iter_per_epoch, i_iter):
    """
    cosine learning rate with warm-up
    """
    if epoch_cur < warmup_epochs:
        # T_cur = 1, 2, 3, ..., (T_total - 1)
        T_cur = 1 + epoch_cur * num_iter_per_epoch + i_iter
        T_total = 1 + warmup_epochs * num_iter_per_epoch
        lr = (T_cur / T_total) * init_lr
    else:
        # T_cur = 0, 1, 2, 3, ..., (T_total - 1)
        T_cur = (epoch_cur - warmup_epochs) * num_iter_per_epoch + i_iter
        T_total = (epoch_total - warmup_epochs) * num_iter_per_epoch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        torch.save(state, os.path.join(save_dir, 'model_best.pth.tar'))


def estimated_time(t_start, cur_epoch, start_epoch, total_epoch):
    t_curr = time.time()
    eta_total = (t_curr - t_start) / (cur_epoch + 1 - start_epoch) * (total_epoch - cur_epoch - 1)
    eta_hour = int(eta_total // 3600)
    eta_min = int((eta_total - eta_hour * 3600) // 60)
    eta_sec = int(eta_total - eta_hour * 3600 - eta_min * 60)
    return f'Finished epoch:{cur_epoch:05d}/{total_epoch:05d};  ETA {eta_hour:02d} h {eta_min:02d} m {eta_sec:02d} s'


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    free_port = sock.getsockname()[1]
    return free_port


""" part of main """
def get_args():
    parser = argparse.ArgumentParser()
    # Model Related
    parser.add_argument("--model_type", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        default="resnet50",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained_models/resnet50-0676ba61.pth",
                        help="Where to search for pretrained models.")
    # Data Related
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "Aircraft", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/home/data')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    # Directory Related
    parser.add_argument("--output_dir_root", default="./", type=str,
                        help="output_dir's root")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    # Optimizer & Learning Schedule
    parser.add_argument("--lr", "--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate.")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Step of training to perform learning rate warmup for.")
    # For a Specific Experiment
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--round', type=int, help="repeat same hyperparameter round")

    # ISDA
    parser.add_argument('--lambda_0', type=float, required=True,
                    help='The hyper-parameter \lambda_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
                         'We adopt 1 for DenseNets and 7.5 for ResNets and ResNeXts, except for using 5 for ResNet-101.')
    # Meta
    parser.add_argument("--meta_lr", default=1e-3, type=float, help="The initial meta learning rate.")
    parser.add_argument('--meta_net_hidden_size', default=512, type=int, required=True)
    parser.add_argument('--meta_net_num_layers', default=1, type=int, required=True)
    parser.add_argument('--meta_weight_decay', type=float, default=0.0)
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def setup_model(args):

    if args.dataset == "CUB_200_2011":
        args.num_classes = 200
    elif args.dataset == "car":
        args.num_classes = 196
    elif args.dataset == "nabirds":
        args.num_classes = 555
    elif args.dataset == "dog":
        args.num_classes = 120
    elif args.dataset == "Aircraft":
        args.num_classes = 100
    elif args.dataset == "INat2017":
        args.num_classes = 5089

    model = eval(args.model_type)(pretrained=True, num_classes=args.num_classes, pretrained_dir=args.pretrained_dir)

    return args, model


def main():
    # Get args
    args = get_args()

    # Setup data_root
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)

    # Setup save path
    args.output_dir = os.path.join(
        args.output_dir_root,
        args.output_dir,
        f'{args.dataset}_{args.model_type}_bs{args.train_batch_size}_lr{args.lr}_wd{args.weight_decay}_epochs{args.epochs}_wmsteps{args.warmup_epochs}_mlr{args.meta_lr}_mhs{args.meta_net_hidden_size}_mlyer{args.meta_net_num_layers}_mdw{args.meta_weight_decay}_lbd{args.lambda_0}_round{args.round}/'
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    set_seed(args)

    # get free port
    args.port = get_free_port()

    # Start Multiprocessing
    mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(), args))


def main_worker(local_rank, ngpus_per_node, args):
    global best_acc1, best_epoch
    args.local_rank = local_rank


    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=os.path.join(args.output_dir, 'screen_output.log'))


    # Multiprocessing
    ip = '127.0.0.1'
    port = args.port
    hosts = 1
    rank = 0
    args.ngpus_per_node = ngpus_per_node
    args.world_size = hosts * args.ngpus_per_node
    args.world_rank = rank * args.ngpus_per_node + args.local_rank
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=args.world_size, rank=args.world_rank)
    args.is_main_proc = (args.world_rank == 0)


    # Model Setup
    args, model = setup_model(args)
    meta_net = MLP_sigmoid(
        input_size=model.feature_num, 
        hidden_size=args.meta_net_hidden_size, 
        num_layers=args.meta_net_num_layers,
        output_size=model.feature_num,
    )


    # DistributedDataParallel
    args.train_batch_size = int(args.train_batch_size / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    meta_net.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    meta_net = torch.nn.parallel.DistributedDataParallel(meta_net, device_ids=[args.local_rank])


    # Prepare optimizer
    criterion_ce = torch.nn.CrossEntropyLoss().cuda(args.local_rank)
    criterion_isda = ISDALoss(model.module.feature_num, args.num_classes, args.local_rank).cuda(args.local_rank)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)


    cudnn.benchmark = True


    # Prepare dataset
    train_loader, test_loader, train_sampler = get_loader(args)


    # Init scores_all.csv
    if args.is_main_proc:
        if not os.path.exists(args.output_dir + '/scores_all.csv'):
            with open(args.output_dir + '/scores_all.csv', "a") as f:
                f.write(f'epoch, lr, loss_train, loss_meta, acc1_train, loss_test, acc1_test, acc1_test_best,\n')


    # Auto Resume
    resume_dir = os.path.join(args.output_dir, "save_models", "checkpoint.pth.tar")
    if os.path.exists(resume_dir):
        logger.info(f'[INFO] resume dir: {resume_dir}')
        ckpt = torch.load(resume_dir, map_location='cpu')
        args.start_epoch = ckpt['epoch']
        model.module.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        meta_net.module.load_state_dict(ckpt['meta_state_dict'])
        meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
        curr_acc1 = ckpt['curr_acc1']
        best_acc1 = ckpt['best_acc1']
        logger.info(f'[INFO] Auto Resume from {resume_dir}, from  finished epoch {args.start_epoch}, with acc_best{best_acc1}, acc_curr {curr_acc1}.')


    # Start Train
    logger.info("***** Running training *****")


    pseudo_net_init = eval(args.model_type)(pretrained=False, num_classes=args.num_classes)


    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        loss_train, loss_meta, acc1_train = train_meta(train_loader, model, meta_net, pseudo_net_init, criterion_isda, criterion_ce, optimizer, meta_optimizer, epoch, args)
        
        if (epoch % 5 == 0) or (epoch >= args.epochs - 15) or (epoch <= 15):
            loss_test, acc1_test = validate(test_loader, model, criterion_ce, args)

            if args.is_main_proc:

                is_best = acc1_test > best_acc1
                best_acc1 = max(acc1_test, best_acc1)
                if is_best:
                    best_epoch = epoch

                with open(args.output_dir + '/scores_all.csv', "a") as f:
                    f.write(
                        f"{epoch:3d}, {get_lr(optimizer):15.12f}, {loss_train:9.8f}, {loss_meta:9.8f}, {acc1_train:6.3f}, {loss_test:9.8f}, {acc1_test:6.3f}, {best_acc1:6.3f},\n"
                    )

                save_checkpoint(
                    {'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'meta_state_dict': meta_net.module.state_dict(),
                    'meta_optimizer': meta_optimizer.state_dict(),
                    'curr_acc1': acc1_test,
                    'best_acc1': best_acc1,
                    'best_epoch': best_epoch,
                    }, is_best, save_dir=os.path.join(args.output_dir, 'save_models')
                )

        if args.is_main_proc:
            logger.info(estimated_time(start_time, epoch, args.start_epoch, args.epochs))

    # record final result
    if args.is_main_proc:
        with open(args.output_dir + '/scores_final.csv', "a") as f:
            f.write(f'epoch, lr, loss_train, loss_meta, acc1_train, loss_test, acc1_test, acc1_test_best,\n')
            f.write(f"{epoch:3d}, {get_lr(optimizer):15.12f}, {loss_train:9.8f}, {loss_meta:9.8f}, {acc1_train:6.3f}, {loss_test:9.8f}, {acc1_test:6.3f}, {best_acc1:6.3f},\n")

    logger.info("Best Accuracy: \t%f" % best_acc1)
    logger.info("Last Accuracy: \t%f" % acc1_test)
    logger.info("Training Complete.")


def train_meta(train_loader, model, meta_net, pseudo_net_init, criterion_isda, criterion_ce, optimizer, meta_optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    meta_losses = AverageMeter('MetaLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, meta_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    pseudo_net_init.cuda(args.local_rank)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)

        images_p1, images_p2 = images.chunk(2, dim=0)
        target_p1, target_p2 = target.chunk(2, dim=0)

        data_time.update(time.time() - end)

        # adjust learning rate
        lr = adjust_learning_rate(optimizer, init_lr=args.lr,
                             epoch_total=args.epochs, warmup_epochs=args.warmup_epochs, epoch_cur=epoch,
                             num_iter_per_epoch=len(train_loader), i_iter=i)
        meta_lr = adjust_meta_learning_rate(meta_optimizer, init_lr=args.meta_lr,
                             epoch_total=args.epochs, warmup_epochs=args.warmup_epochs, epoch_cur=epoch,
                             num_iter_per_epoch=len(train_loader), i_iter=i)
        ratio = args.lambda_0 * (epoch / args.epochs)

        ###################################################
        ## part 1: images_p1 as train, images_p2 as meta ##
        ###################################################
        pseudo_net = pickle.loads(pickle.dumps(pseudo_net_init))
        pseudo_net.load_state_dict(model.module.state_dict())
        pseudo_net.train()

        pseudo_outputs_features = pseudo_net(images_p1)
        pseudo_cv_matrix = meta_net(pseudo_outputs_features.detach())
        pseudo_loss, pseudo_outputs_logits = criterion_isda(pseudo_net.head, pseudo_outputs_features, target_p1, ratio, pseudo_cv_matrix)

        pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True, allow_unused=True)

        pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
        pseudo_optimizer.load_state_dict(optimizer.state_dict())
        pseudo_optimizer.meta_step(pseudo_grads)

        del pseudo_grads


        meta_outputs_features = pseudo_net(images_p2)
        meta_outputs_logits = pseudo_net.head(meta_outputs_features)
        meta_loss = criterion_ce(meta_outputs_logits, target_p2)
        meta_losses.update(meta_loss.item(), images_p2.size(0))

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()


        outputs_features = model(images_p1)
        cv_matrix = meta_net(outputs_features)
        loss, outputs_logits = criterion_isda(model.module.head, outputs_features, target_p1, ratio, cv_matrix)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs_logits, target_p1, topk=(1, 5))
        losses.update(loss.item(), images_p1.size(0))
        top1.update(acc1[0].item(), images_p1.size(0))
        top5.update(acc5[0].item(), images_p1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        ###################################################
        ## part 2: images_p2 as train, images_p1 as meta ##
        ###################################################
        pseudo_net = pickle.loads(pickle.dumps(pseudo_net_init))
        pseudo_net.load_state_dict(model.module.state_dict())
        pseudo_net.train()

        pseudo_outputs_features = pseudo_net(images_p2)
        pseudo_cv_matrix = meta_net(pseudo_outputs_features.detach())
        pseudo_loss, pseudo_outputs_logits = criterion_isda(pseudo_net.head, pseudo_outputs_features, target_p2, ratio, pseudo_cv_matrix)

        pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True, allow_unused=True)

        pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
        pseudo_optimizer.load_state_dict(optimizer.state_dict())
        pseudo_optimizer.meta_step(pseudo_grads)

        del pseudo_grads


        meta_outputs_features = pseudo_net(images_p1)
        meta_outputs_logits = pseudo_net.head(meta_outputs_features)
        meta_loss = criterion_ce(meta_outputs_logits, target_p1)
        meta_losses.update(meta_loss.item(), images_p1.size(0))

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()


        outputs_features = model(images_p2)
        cv_matrix = meta_net(outputs_features)
        loss, outputs_logits = criterion_isda(model.module.head, outputs_features, target_p2, ratio, cv_matrix)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs_logits, target_p2, topk=(1, 5))
        losses.update(loss.item(), images_p2.size(0))
        top1.update(acc1[0].item(), images_p2.size(0))
        top5.update(acc5[0].item(), images_p2.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ###################################################
        ##                finish exchange                ##
        ###################################################


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i % args.print_freq == 0) or (i == len(train_loader) - 1)) and args.is_main_proc:
            progress.display(i)
    
    return losses.avg, meta_losses.avg, top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)

            # compute output
            features = model(images)
            logits = model.module.head(features)            
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            dist.all_reduce(acc1)
            acc1 /= args.world_size
            dist.all_reduce(acc5)
            acc5 /= args.world_size
            dist.all_reduce(loss)
            loss /= args.world_size

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i % args.print_freq == 0) or (i == len(val_loader) - 1)) and args.is_main_proc:
                progress.display(i)
        
        if args.is_main_proc:
            progress.display_summary()

    return losses.avg, top1.avg


if __name__ == '__main__':
    main()