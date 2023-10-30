import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

## modified from https://github.com/Lingkai-Kong/SDE-Net

def SVHN(download=True, transform=transforms.ToTensor(), **kwargs):
    data_path = os.path.join(os.getcwd(), 'data', 'svhn')
    d1 = datasets.SVHN(root=data_path, split='train', download=True, transform=transform)
    d2 = datasets.SVHN(root=data_path, split='test', download=True, transform=transform)
    dataset = torch.utils.data.ConcatDataset([d1, d2])
    return dataset

# def get_mean_std(loader):
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#     for data, _ in loader:
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1
#     mean = channels_sum / num_batches
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
#     return mean, std

# def getSVHN(batch_size, img_size, **kwargs):
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     print("Building SVHN data loader with {} workers".format(num_workers))

#     def target_transform(target):
#         new_target = target - 1
#         if new_target == -1:
#             new_target = 9
#         return new_target

#     train_loader = DataLoader(
#         datasets.SVHN(
#             root='../data/svhn', split='train', download=True,
#             transform=transforms.Compose([
#                 transforms.Resize(img_size),
#                 transforms.ToTensor(),
#             ]),
#             target_transform=target_transform,
#         ),
#         batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

#     return train_loader


# if __name__ == '__main__':
    # d1 = datasets.SVHN(root='../data/svhn', split='train', download=True, transform=transforms.ToTensor())
    # d2 = datasets.SVHN(root='../data/svhn', split='test', download=True, transform=transforms.ToTensor())
    # dataset = torch.utils.data.ConcatDataset([d1, d2])
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    # mean, std = get_mean_std(dataloader)
    # print(mean)
    # print(std)
    # train_loader = getSVHN(1000, 28)
    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    #     print(inputs.shape)
    #     print(targets.shape)
