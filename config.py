############### Pytorch CIFAR configuration file ###############
import math


batch_size = 128
optim_type = 'AdamW'

mean = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'cifar10_clip': [0.4914, 0.4822, 0.4465],
    'cifar10_lt': [0.4914, 0.4822, 0.4465],
    # 'cifar100': (0.5071, 0.4865, 0.4409),
    'cifar100': [0.5071, 0.4867, 0.4408],
    'cifar100_clip': [0.5071, 0.4867, 0.4408],
    'cifar100_lt': [0.5071, 0.4867, 0.4408],
    'ti': (0.485, 0.456, 0.406),
    'mnist': (0.1307,),
    'svhn': (0.4416, 0.4461, 0.4718),
    'lsun': (0.5084, 0.4706, 0.4341),
    'webvision':(0.485, 0.456, 0.406),
}

std = {
    'cifar10': [0.2023, 0.1994, 0.2010],
    'cifar10_clip': [0.2023, 0.1994, 0.2010],
    'cifar10_lt': [0.2023, 0.1994, 0.2010],
    # 'cifar100': (0.2673, 0.2564, 0.2762),
    'cifar100': [0.2675, 0.2565, 0.2761],
    'cifar100_clip': [0.2675, 0.2565, 0.2761],
    'cifar100_lt': [0.2675, 0.2565, 0.2761],
    'ti': (0.229, 0.224, 0.225),
    'mnist': (0.3081,),
    'svhn': (0.2040, 0.2081, 0.2058),
    'lsun': (0.2487, 0.2492, 0.2675),
    'webvision':(0.229, 0.224, 0.225),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

DEVICES = '0,1,2,3'

