# The implement of 'Towards Accelerated Model Training via Bayesian Data Selection'

Thank you for taking the time to review our code and datasets. This readme describes how to run our proposed method. Notice: Our programs are build based on GPU, it is better to test them in GPU.

## Environment Settings

- numpy==1.20.1
- scikit_learn==1.2.1
- scipy==1.6.2
- timm==0.5.4
- torch==2.0.1
- torchvision==0.15.1
- tqdm==4.59.0
- transformers==4.18.0

## How to run

After you have downloaded the repository, you can train the model using Bayesian data selection by running the example script below.


* For CIFAR-10

```bash
CUDA_VISIBLE_DEVICES=0 python clip_prioritized_train_bayes_ema.py --num_epochs 200 --dataset cifar10_clip --save_name bayes_e200_tau4_alpha.2_2e2d_ema --alpha .2 --num_effective_data 200 --prior_precision 10 --tau 4
```


* For CIFAR-100

```bash
CUDA_VISIBLE_DEVICES=0 python clip_prioritized_train_bayes_ema.py --num_epochs 200 --dataset cifar100_clip --save_name bayes_e200_ema --alpha .3 --tau 12 --num_effective_data 1000 --prior_precision 10
```

* For WebVision-100

```bash
CUDA_VISIBLE_DEVICES=0 python clip_prioritized_train_bayes_ema.py --num_epochs 200 --dataset webvision --num_classes 100 --save_name bayes_e200_ema --alpha .3 --tau 10 --num_effective_data 400 --prior_precision 10
```