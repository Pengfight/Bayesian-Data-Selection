import torch
import torch.nn as nn
from torch.nn.functional import dropout
import torchvision
from models import resnet_cifar, resnet_cifar_dropout

# class Ensemble(nn.Module):
#     def __init__(self, ensemble_num, num_classes=10, dataset='cifar10',dropout=False):
#         super(Ensemble, self).__init__()
#         if dataset != 'imagenet':
#             if dropout:
#                 self.nets = nn.ModuleList([resnet_cifar_dropout.ResNet18(num_classes) for _ in range(ensemble_num)])
#             else:
#                 self.nets = nn.ModuleList([resnet_cifar.ResNet18(num_classes) for _ in range(ensemble_num)])
#         else:
#             self.nets = nn.ModuleList([torchvision.models.resnet18(num_classes=num_classes) for _ in range(ensemble_num)])
#         self.ensemble_num = ensemble_num


#     def forward(self, x):
#         return tuple(net(x) for net in self.nets)

class Ensemble(nn.Module):
    def __init__(self, ensemble_num, net_fn, model_args):
        super(Ensemble, self).__init__()
        self.nets = nn.ModuleList([net_fn(*model_args) for _ in range(ensemble_num)])
        self.ensemble_num = ensemble_num


    def forward(self, x):
        return tuple(net(x) for net in self.nets)