import numpy as np
import pandas as pd
import torch
import torch.utils.data as td
import torchvision.datasets as tvd
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as tvt
import sklearn.datasets as skd

import os
curr_dir = os.getcwd()
DATASET_DIR = '/data/LargeData/Regular/cifar'

def get_dataset(dataset_name, batch_size, split=[.8, .2], seed=0, test_shuffle_seed=None, batch_size_eval=1024, n_test_data=None):
    
    if dataset_name == 'MNIST':
        dataset = Dataset.mnist(train=True)
        test_set = Dataset.mnist(train=False)
        output_dim = 10
    elif dataset_name == 'KMNIST':
        dataset = Dataset.kmnist(train=True)
        test_set = Dataset.kmnist(train=False)
        output_dim = 10
    elif dataset_name == 'FMNIST':
        dataset = Dataset.fashion_mnist(train=True)
        test_set = Dataset.fashion_mnist(train=False)
        output_dim = 10
    elif dataset_name == 'CIFAR10':
        dataset = Dataset.cifar10(train=True, image_transforms=[tvt.RandomHorizontalFlip(),
                                                                tvt.RandomCrop(32, 4),
                                                                tvt.RandomRotation(degrees=15)])
        test_set = Dataset.cifar10(train=False)
        output_dim = 10
    elif dataset_name == 'CIFAR10_oodom':
        dataset = Dataset.cifar10(train=True,
                                  image_transforms=[tvt.RandomHorizontalFlip(),
                                                    tvt.RandomCrop(32, 4),
                                                    tvt.RandomRotation(degrees=15)],
                                  tensor_transforms=[tvt.Lambda(lambda x: x * 255.)])
        test_set = Dataset.cifar10(train=False, tensor_transforms=[tvt.Lambda(lambda x: x * 255.)])
        output_dim = 10
    elif dataset_name == 'CIFAR100':
        dataset = Dataset.cifar100(train=True, image_transforms=[tvt.RandomHorizontalFlip(),
                                                                tvt.RandomCrop(32, 4),
                                                                tvt.RandomRotation(degrees=15)])
        test_set = Dataset.cifar100(train=False)
        output_dim = 100
    elif dataset_name == 'CIFAR100_oodom':
        dataset = Dataset.cifar100(train=True,
                                   image_transforms=[tvt.RandomHorizontalFlip(),
                                                     tvt.RandomCrop(32, 4),
                                                     tvt.RandomRotation(degrees=15)],
                                   tensor_transforms=[tvt.Lambda(lambda x: x * 255.)])
        test_set = Dataset.cifar100(train=False, tensor_transforms=[tvt.Lambda(lambda x: x * 255.)])
        output_dim = 100
    elif dataset_name == 'SVHN':
        dataset = Dataset.svhn(train=True)
        test_set = Dataset.svhn(train=False)
        output_dim = 10
    elif dataset_name == 'SVHN_oodom':
        dataset = Dataset.svhn(train=True, tensor_transforms=[tvt.Lambda(lambda x: x * 255.)])
        test_set = Dataset.svhn(train=False, tensor_transforms=[tvt.Lambda(lambda x: x * 255.)])
        output_dim = 10
    elif dataset_name == 'LSUN':
        dataset = Dataset.svhn(train=True)
        test_set = Dataset.svhn(train=False)
        output_dim = 10
    elif dataset_name == 'FAKE':
        dataset = Dataset.fake(train=True)
        test_set = Dataset.fake(train=False)
        output_dim = 10
    else:
        raise NotImplementedError

    indices = list(range(len(dataset)))
    assert np.sum(split) == 1.0
    split = int(len(dataset) * split[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_eval, sampler=val_sampler, num_workers=0)
    if n_test_data is not None:
        test_set = torch.utils.data.Subset(test_set, np.arange(min(n_test_data, len(test_set))))
    test_loader = td.DataLoader(test_set, batch_size_eval, shuffle=True, num_workers=0)

    N = torch.zeros(output_dim, dtype=torch.long)
    for _, Y in train_loader:
        N.scatter_add_(0, Y, torch.ones_like(Y))

    return train_loader, val_loader, test_loader, N


class Dataset:
    """
    The dataset class provides static methods to use different datasets and their splits for
    different seeds. The following is ensured for all datasets:
    * All features are normalized to have zero mean and unit variance.
    All static methods accept a single `seed` parameter which governs the seed to use for splitting
    the data into train/val/test. They always return a triple with PyTorch datasets for training,
    validation, and test data, respectively.
    """

    @classmethod
    def toy_classification(cls):
        """
        Generates a 2D toy dataset consisting of three clusters to be used for classification. Each
        cluster has its own label, two clusters are slightly overlapping to model a region of high
        epistemic uncertainty.
        * Task: classification
        * Features: [2]
        * Samples: 3,072
        """
        X, y = skd.make_blobs(
            3072, 2, centers=[(-1, 0), (1, 0), (0.5, 0.5)], cluster_std=0.25, random_state=42
        )
        return cls._tensor_dataset(X.astype(np.float32), y.astype(np.int64))

    @classmethod
    def segment_window_sky_missing(cls, train=True):
        """
        Returns the segment dataset, either for training or testing.
        * Task: classification
        * Features: [18]
        * Samples: 1650
        """

        directory_dataset = '/nfs/staff-hdd/charpent/dirichlet-robustness/datasets/segmentation/'
        values = pd.read_csv(directory_dataset + 'segment_window_sky_missing.csv', header=0, index_col=0).values
        n_data, input_dim = values.shape[0], values.shape[1] - 1

        # Features and label data
        if train:
            X, y = values[:int(.8 * n_data), :-1], np.squeeze(values[:int(.8 * n_data), -1:])
        else:
            X, y = values[int(.8 * n_data):, :-1], np.squeeze(values[int(.8 * n_data):, -1:])
        return cls._tensor_dataset(X.astype(np.float32), y.astype(np.int64))

    @classmethod
    def segment_window_only(cls):
        """
        Returns the segment dataset with class window.
        * Task: classification
        * Features: [18]
        * Samples: 330
        """

        directory_dataset = '/nfs/staff-hdd/charpent/dirichlet-robustness/datasets/segmentation/'
        values = pd.read_csv(directory_dataset + 'segment_window_only.csv', header=0, index_col=0).values
        n_data, input_dim = values.shape[0], values.shape[1] - 1

        # Features and label data
        X, y = values[:, :-1], np.squeeze(values[:, -1:])
        return cls._tensor_dataset(X.astype(np.float32), y.astype(np.int64))

    @classmethod
    def segment_sky_only(cls):
        """
        Returns the segment dataset with class sky.
        * Task: classification
        * Features: [18]
        * Samples: 330
        """

        directory_dataset = '/nfs/staff-hdd/charpent/dirichlet-robustness/datasets/segmentation/'
        values = pd.read_csv(directory_dataset + 'segment_sky_only.csv', header=0, index_col=0).values
        n_data, input_dim = values.shape[0], values.shape[1] - 1

        # Features and label data
        X, y = values[:, :-1], np.squeeze(values[:, -1:])
        return cls._tensor_dataset(X.astype(np.float32), y.astype(np.int64))

    @classmethod
    def sensorless_drive_9_10_11_missing(cls, train=True):
        """
        Returns the segment dataset, either for training or testing.
        * Task: classification
        * Features: [48]
        * Samples: 42552
        """

        directory_dataset = '/nfs/staff-hdd/charpent/dirichlet-robustness/datasets/sensorless_drive/'
        values = pd.read_csv(directory_dataset + 'sensorless_drive_9_10_11_missing.csv', header=0, index_col=0).values
        n_data, input_dim = values.shape[0], values.shape[1] - 1

        # Features and label data
        if train:
            X, y = values[:int(.8 * n_data), :-1], np.squeeze(values[:int(.8 * n_data), -1:])
        else:
            X, y = values[int(.8 * n_data):, :-1], np.squeeze(values[int(.8 * n_data):, -1:])
        return cls._tensor_dataset(X.astype(np.float32), y.astype(np.int64))

    @classmethod
    def sensorless_drive_9_only(cls):
        """
        Returns the segment dataset with class 9.
        * Task: classification
        * Features: [48]
        * Samples: 5319
        """

        directory_dataset = '/nfs/staff-hdd/charpent/dirichlet-robustness/datasets/sensorless_drive/'
        values = pd.read_csv(directory_dataset + 'sensorless_drive_9_only.csv', header=0, index_col=0).values
        n_data, input_dim = values.shape[0], values.shape[1] - 1

        # Features and label data
        X, y = values[:, :-1], np.squeeze(values[:, -1:])
        return cls._tensor_dataset(X.astype(np.float32), y.astype(np.int64))

    @classmethod
    def sensorless_drive_10_11_only(cls):
        """
        Returns the segment dataset wiht class 10 and 11.
        * Task: classification
        * Features: [48]
        * Samples: 10638
        """

        directory_dataset = '/nfs/staff-hdd/charpent/dirichlet-robustness/datasets/sensorless_drive/'
        values = pd.read_csv(directory_dataset + 'sensorless_drive_10_11_only.csv', header=0, index_col=0).values
        n_data, input_dim = values.shape[0], values.shape[1] - 1

        # Features and label data
        X, y = values[:, :-1], np.squeeze(values[:, -1:])
        return cls._tensor_dataset(X.astype(np.float32), y.astype(np.int64))


    @classmethod
    def mnist(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the MNIST dataset, either for training or testing:
        * Task: classification
        * Features: [1, 28, 28]
        * Classes: 10
        * Samples: 60,000 | 10,000
        """
        return tvd.MNIST(
            DATASET_DIR, download=True, train=train,
            transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms)
        )

    @classmethod
    def kmnist(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the KMNIST dataset, either for training or testing:
        * Task: classification
        * Features: [1, 28, 28]
        * Classes: 10
        * Samples: 60,000 | 10,000
        """
        return tvd.KMNIST(
            DATASET_DIR, download=True, train=train,
            transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms)
        )

    @classmethod
    def fashion_mnist(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the Fashion-MNIST dataset, either for training or testing:
        * Task: classification
        * Features: [1, 28, 28]
        * Classes: 10
        * Samples: 60,000 | 10,000
        """
        return tvd.FashionMNIST(
            DATASET_DIR, download=True, train=train,
            transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms)
        )

    @classmethod
    def cifar10(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the CIFAR-10 dataset, either for training or testing:
        * Task: classification
        * Features: [3, 32, 32]
        * Classes: 10
        * Samples: 50,000 | 10,000
        """
        return tvd.CIFAR10(
            DATASET_DIR, download=True, train=train,
            transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms)
        )

    @classmethod
    def cifar100(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the CIFAR-100 dataset, either for training or testing:
        * Task: classification
        * Features: [3, 32, 32]
        * Classes: 10
        * Samples: 50,000 | 10,000
        """
        return tvd.CIFAR100(
            DATASET_DIR, download=True, train=train,
            transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms)
        )

    @classmethod
    def svhn(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the SVHN dataset, either for training or testing:
        * Task: classification
        * Features: [3, 32, 32]
        * Classes: 10
        * Samples: 73,257 | 26,032
        """
        split = 'train' if train else 'test'
        return tvd.SVHN(
            DATASET_DIR, download=True, split=split,
            transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms)
        )

    @classmethod
    def lsun(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the SVHN dataset, either for training or testing:
        * Task: classification
        * Features: XXX
        * Classes: 10
        * Samples: 73,257 | 26,032
        """
        split = 'train' if train else 'test'
        return tvd.LSUN(DATASET_DIR, classes=split,
                        transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms, target_transform=None))

    @classmethod
    def fake(cls, train=True, image_transforms=[], tensor_transforms=[]):
        """
        Returns the SVHN dataset, either for training or testing:
        * Task: classification
        * Features: [3, 32, 32]
        * Classes: 10
        * Samples: 100 000
        """
        return tvd.FakeData(size=100000, image_size=(3, 32, 32), num_classes=10,
                            transform=tvt.Compose(image_transforms + [tvt.ToTensor()] + tensor_transforms))

    @classmethod
    def random_noise_image_dataset(cls, num_classes, num_images_per_class, mean=0, sigma=1, dims=(1, 28, 28),
                                   bounds=None):
        if bounds is None:
            bounds = torch.tensor([0.0, 1.0], dtype=torch.float32)
        clip_lower, clip_upper = bounds
        import random
        random_ixs = list(range(num_classes*num_images_per_class))
        random.shuffle(random_ixs)

        generate_dims = (num_classes*num_images_per_class,) + dims
        X = torch.randn(generate_dims) * sigma + mean
        X = X.clamp(clip_lower, clip_upper)
        y = torch.repeat_interleave(torch.arange(0, num_classes), num_images_per_class)

        return td.TensorDataset(X[random_ixs], y[random_ixs])