import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

default_transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ])

def LSUN(transform=default_transform, **kwargs):
    # scriptdir = os.path.dirname(__file__)
    # data_path = os.path.join(scriptdir,'data')
    data_path = os.path.join(os.getcwd(), 'data', 'lsun')
    return datasets.LSUN(root=data_path, classes='test', transform=transform)

# def get_mean_std(loader):
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#     for data, _ in loader:
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1
#     mean = channels_sum / num_batches
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
#     return mean, std

if __name__ == "__main__":
    trans = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ])
    lsun_testset = datasets.LSUN(root='../data/lsun', classes='test', transform=trans)
    dataloader = DataLoader(lsun_testset, batch_size=64, shuffle=False)
    # mean, std = get_mean_std(dataloader)
    # print(mean)
    # print(std)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(inputs.shape)
        print(targets.shape)
        sys.exit(0)
