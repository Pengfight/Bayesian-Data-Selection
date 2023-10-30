from PIL import Image
import os
import os.path
import numpy as np
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import *
from torchvision.datasets.mnist import *
from urllib.error import URLError
import torchvision
from math import *

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from .utils import noisify_x, noisify_overlap, noisify_y, add_trigger


cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck',]
cifar10_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

cifar100_classes = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]

cifar100_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


class CIFAR10Noise_clip(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    nb_classes = 10
    template = cifar10_templates
    classnames = cifar10_classes
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str = '/data/LargeData/Regular/cifar',
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            xnoise_type='gaussian', xnoise_arg=1, xnoise_rate=0, 
            ynoise_type='symmetric', ynoise_rate=0,
            trigger_size=3, trigger_rate=0.0,
            random_state=0
    ) -> None:

        super(CIFAR10Noise_clip, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        if self.train:
            # if trigger_rate > 0.0:
            #     self.data, self.targets, self.poison_or_not= add_trigger(trainset=self.data, targets = self.targets, trigger_size=trigger_size, trigger_rate=trigger_rate, random_state=random_state)
            # else:
            #     self.poison_or_not = np.zeros(self.data.shape[0])
            if xnoise_rate > 0.0:
                self.data, self.xnoisy_or_not= noisify_x(dataset='cifar', data=self.data, noise_type=xnoise_type, noise_arg=xnoise_arg, noise_rate=xnoise_rate, random_state=random_state)
            else:
                self.xnoisy_or_not = np.zeros(self.data.shape[0])
            self.xnoisy_or_not = (self.xnoisy_or_not == 1)
            if ynoise_rate > 0.0:
                self.targets = np.asarray([[self.targets[i]] for i in range(len(self.targets))])
                # self.noise_targets = noisify_y(train_labels=self.targets, noise_type=ynoise_type, noise_rate=ynoise_rate, random_state=random_state, nb_classes=self.nb_classes,xnoisy_or_not=self.xnoisy_or_not)
                self.noise_targets = noisify_y(train_labels=self.targets, noise_type=ynoise_type, noise_rate=ynoise_rate, random_state=random_state, nb_classes=self.nb_classes)
                self.noise_targets = np.asarray([i[0] for i in self.noise_targets])
                self.targets =  np.asarray([i[0] for i in self.targets])
                self.ynoisy_or_not = (self.targets != self.noise_targets)
            else:
                self.noise_targets = self.targets
                self.ynoisy_or_not = np.zeros(self.data.shape[0]).astype(np.int64)
            self.report_noise()
        self._load_meta()

    def get_noise(self): 
        xy_noise = np.logical_and(self.xnoisy_or_not, self.ynoisy_or_not)
        x_noise = np.logical_and(self.xnoisy_or_not, ~self.ynoisy_or_not)
        y_noise = np.logical_and(~self.xnoisy_or_not, self.ynoisy_or_not)
        # poison = np.logical_and(self.poison_or_not, ~self.xnoisy_or_not)
        clean = np.logical_and(~self.xnoisy_or_not, ~self.ynoisy_or_not)
        return {
            'xy_noise': xy_noise,
            'x_noise': x_noise,
            'y_noise': y_noise,
            # 'poison': poison,
            'clean': clean,
            'xnoisy': self.xnoisy_or_not,
            'ynoisy': self.ynoisy_or_not
        }
    def report_noise(self):
        noise_stat = self.get_noise()
        print('Noise Stat:')
        for key, val in noise_stat.items():
            print(key, val.sum())
            
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, noise_tar, true_tar, xnoisy = self.data[index], int(self.noise_targets[index]), int(self.targets[index]), self.xnoisy_or_not[index]
        else:
            img, target = self.data[index], int(self.targets[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        if self.train:
            return ((img, xnoisy), (noise_tar, true_tar))
        else:
            return (img, target)


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



class CIFAR100Noise_clip(CIFAR10Noise_clip):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    nb_classes = 100
    template = cifar100_templates
    classnames = cifar100_classes

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    nb_classes = 10
    template = cifar10_templates
    classnames = cifar10_classes
    decay_stride = 2.1971

    def __init__(self,root: str = '/data/LargeData/Regular/cifar',imb_type='exp',train=True,transform=None,target_transform=None,download=False):
        super(IMBALANCECIFAR10,self).__init__(root, train, transform, target_transform, download)
        if train:
            img_num_list = self.get_img_num_per_cls(self.nb_classes,imb_type)
            self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self,num_class,imb_type):
        img_max = len(self.data)/num_class
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(num_class):
                num = img_max*exp(-cls_idx/self.decay_stride)
                img_num_per_cls.append(int(num+0.5))
        else:
            img_num_per_cls.extend([int(img_max)]*num_class)
        return img_num_per_cls

    def gen_imbalanced_data(self,img_num_per_cls):
        img_max = len(self.data)/self.nb_classes
        new_data,new_targets = [],[]
        targets_np = np.array(self.targets,dtype=np.int64)
        classes = np.arange(self.nb_classes)

        self.num_per_cls = np.zeros(self.nb_classes)
        for class_i,volume_i in zip(classes,img_num_per_cls):
            self.num_per_cls[class_i] = volume_i
            idx = np.where(targets_np==class_i)[0]
            np.random.shuffle(idx)
            keep_num = volume_i+1
            selec_idx = idx[:keep_num]
            new_data.append(self.data[selec_idx,...])
            new_targets.extend([class_i]*keep_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        return self.num_per_cls.tolist()

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    nb_classes = 100
    decay_stride = 21.9714
    template = cifar100_templates
    classnames = cifar100_classes
    

if __name__ == '__main__':
    dataset = CIFAR10Noise_clip( train=True, xnoise_rate=0.1, ynoise_rate=0.1)
    print(dataset.get_noise())