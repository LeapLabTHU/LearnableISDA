# Fine-grained Recognition with Learnable Semantic Data Augmentation

accepted by IEEE Transactions on Image Processing (IEEE TIP)

Authors: [Yifan Pu](https://github.com/yifanpu001/)\*, [Yizeng Han](https://yizenghan.top/)\*, [Yulin Wang](https://www.wyl.cool/), [Junlan Feng](https://scholar.google.com/citations?user=rBjPtmQAAAAJ&hl=en&oi=ao), Chao Deng, [Gao Huang](http://www.gaohuang.net/)\#.

*: Equal contribution, #: Corresponding author.


## Get Started
1. prepare environment
    ```
    conda create --name learnable_isda python=3.8
    conda activate learnable_isda
    pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
    pip install scipy pandas matplotlib imageio
    ```

2. prepare data

    Download CUB-200-2011 from the [official website](https://www.vision.caltech.edu/datasets/cub_200_2011/)

3. prepare pretrained checkpoint
    ```
    mkdir pretrained_models
    cd pretrained_models
    wget https://download.pytorch.org/models/resnet50-0676ba61.pth
    cd ..
    ```

## Usage

training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
--data_root YOUR_DATA_PATH --output_dir_root ./ --output_dir output/ \
--model_type resnet50 --pretrained_dir ./pretrained_models/resnet50-0676ba61.pth \
--dataset CUB_200_2011 --train_batch_size 128 --lr 3e-2 --eval_batch_size 64 --workers 1 \
--meta_lr 1e-3 --meta_net_hidden_size 512 --meta_net_num_layers 1 --lambda_0 10.0 \
--epochs 100 --warmup_epochs 5;
```

## Citation

If you find our work is useful in your research, please consider citing:

```
@article{pu2023fine,
  title={Fine-grained recognition with learnable semantic data augmentation},
  author={Pu, Yifan and Han, Yizeng and Wang, Yulin and Feng, Junlan and Deng, Chao and Huang, Gao},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}
```

## Contact
If you have any questions, please feel free to contact the authors.

Yifan Pu: pyf20@mails.tsinghua.edu.cn, yifanpu98@126.com.

Yizeng Han: hanyz18@mails.tsinghua.edu.cn, yizeng38@gmail.com.
