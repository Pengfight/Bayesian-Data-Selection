from __future__ import print_function
import os

from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from torch import nn
import os, sys
import copy

import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import config as cf 
from datasets import CIFAR10Noise, CIFAR10Noise_clip, CIFAR100Noise, CIFAR100Noise_clip, IMBALANCECIFAR10, IMBALANCECIFAR100, TinyImagenetNoise, SVHN, LSUN
from networks import *
import clip

def ECELoss(logits, labels, n_bins = 15):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def build_dataset(is_train, args):
    # transform = build_transform(is_train, args)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(cf.mean[args.data_set.lower()], cf.std[args.data_set.lower()])                
                                      
    transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform)
        nb_classes = 10
    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def get_transforms(train=True, dataset='cifar10'):
    if dataset == 'ti':
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
            ]) # meanstd transformation
        else:
            return transforms.Compose([transforms.Resize(64),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
            ])
    elif dataset == "lsun":
        return transforms.Compose([
            # transforms.Resize([32, 32]),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
        ])
    else:
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
            ]) # meanstd transformation
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cf.mean[dataset], cf.std[dataset]),
            ])

DSETS = {
    'cifar10': CIFAR10Noise,
    'cifar10_clip': CIFAR10Noise_clip,
    'cifar100_clip': CIFAR100Noise_clip,
    'cifar100': CIFAR100Noise,
    'cifar10_lt': IMBALANCECIFAR10,
    'cifar100_lt': IMBALANCECIFAR100,
    'ti': TinyImagenetNoise,
    'svhn': SVHN,
    'lsun': LSUN,
}
def prepare_dset(args):
    transform_train = get_transforms(train=True, dataset=args.dataset)
    transform_test = get_transforms(train=False, dataset=args.dataset)
    DSET = DSETS[args.dataset]
    print(f"| Preparing {args.dataset} dataset with noisy img...")
    trainset = DSET(
                    download=True,
                    train=True,
                    transform=transform_train,
                    xnoise_type=args.xnoise_type,
                    xnoise_rate=args.xnoise_rate,
                    xnoise_arg=args.xnoise_arg,
                    ynoise_type=args.ynoise_type,
                    ynoise_rate=args.ynoise_rate,
                    # trigger_size = args.trigger_size,
                    # trigger_rate = args.trigger_ratio,
                    random_state=args.random_state
                    )
    testset = DSET(
                    download=True,
                    train=False,
                    transform=transform_test)
    trainvalset = copy.deepcopy(trainset)
    trainvalset.transform = transform_test # no data aug
    return trainset, testset, trainvalset

def prepare_dset_lt(args):
    transform_train = get_transforms(train=True, dataset=args.dataset)
    transform_test = get_transforms(train=False, dataset=args.dataset)
    DSET = DSETS[args.dataset]
    trainset = DSET(train=True,download=True,transform=transform_train)
    testset = DSET(train=False,download=True,transform=transform_test)
    return trainset, testset

def prepare_dset_multi(args):
    transform_train = get_transforms(train=True, dataset=args.dataset)
    transform_test = get_transforms(train=False, dataset=args.dataset)
    trainset = TinyImagenetNoiseMulti(
                    download=True,
                    train=True,
                    transform=transform_train,
                    xnoise_types=args.xnoise_types,
                    xnoise_rates=args.xnoise_rates,
                    xnoise_args=args.xnoise_args,
                    ynoise_type=args.ynoise_type,
                    ynoise_rate=args.ynoise_rate,
                    random_state = args.random_state
                    )
    testset = TinyImagenetNoise(
                    download=True,
                    train=False,
                    transform=transform_test)
    trainvalset = copy.deepcopy(trainset)
    trainvalset.transform = transform_test # no data aug
    return trainset, testset, trainvalset

def prepare_dset_large(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]) 
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])  
    if args.dataset == 'webvision':
        train_dataset = webvision_dataset(transform=transform_train, mode="all", num_class=50)
        val_dataset =  webvision_dataset(transform=transform_test, mode="test", num_class=50)    
    else:
        raise NotImplementedError
   
    return train_dataset, val_dataset

def prepare_dset_test(dataset):
    print(f"| Preparing {dataset}")
    if dataset == "gaussian_noise":
        # 10000 samples
        # return TensorDataset(torch.randn(10000, 3, 32, 32), torch.ones(10000))
        return TensorDataset(torch.randn(10000, 3, 224, 224), torch.ones(10000))
    elif dataset == "uniform_noise":
        # 10000 samples
        return TensorDataset(torch.rand(10000, 3, 32, 32), torch.ones(10000))
        return TensorDataset(torch.rand(10000, 3, 224, 224), torch.ones(10000))
    elif dataset == "ti":
        transform_test = transforms.Compose([transforms.Resize(32),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cf.mean[dataset], cf.std[dataset]),])
    else:
        transform_test = get_transforms(train=False, dataset=dataset)
    DSET = DSETS[dataset]
    testset = DSET(download=True,
                   train=False,
                   transform=transform_test)
    return testset
    
def adjust_learning_rate(optimizer, epoch, init_lr, steps):
    """Sets the learning rate"""
    lr = init_lr
    for step in steps:
        if epoch > step: lr = lr * 0.1 
        else: break 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def update_print(s):
    sys.stdout.write('\r')
    sys.stdout.write(s)
    sys.stdout.flush()

class AverageMeter():
    def __init__(self):
        self.sum = 0.
        self.cnt = 0
        self.history = []
    def append(self,x):
        self.history += list(x.cpu().numpy())
        self.cnt += x.shape[0]
        self.sum += x.sum().cpu().item()
    def get(self):
        return self.sum / self.cnt
    
def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights