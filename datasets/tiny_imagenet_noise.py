import numpy as np 
import os 
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from .utils import noisify_x, noisify_overlap, noisify_y


class TinyImagenetNoise(VisionDataset):
    nb_classes = 200
    def __init__(
            self,
            root: str = '/data/LargeData/Regular/tinyimagenet',
            train: bool = True,
            transform= None,
            target_transform = None,
            download: bool = False,
            xnoise_type='gaussian', xnoise_arg=1, xnoise_rate=0, 
            ynoise_type='symmetric', ynoise_rate=0,
            random_state=0
    ) -> None:

        super(TinyImagenetNoise, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train 
        if self.train:
            obj = np.load(os.path.join(root, 'train.npz'))
        else:
            obj = np.load(os.path.join(root, 'test.npz'))
        
        self.data, self.targets = obj['X'], obj['Y']
        self.data = np.uint8(self.data)

        if self.train:
            if xnoise_rate > 0.0:
                self.data, self.xnoisy_or_not= noisify_x(dataset='ti', data=self.data, noise_type=xnoise_type, noise_arg=xnoise_arg, noise_rate=xnoise_rate, random_state=random_state)
            else:
                self.xnoisy_or_not = np.zeros(self.data.shape[0])
            self.xnoisy_or_not = (self.xnoisy_or_not == 1)
            if ynoise_rate > 0.0:
                self.targets = np.asarray([[int(self.targets[i])] for i in range(len(self.targets))])
                # self.noise_targets = noisify_y(train_labels=self.targets, noise_type=ynoise_type, noise_rate=ynoise_rate, random_state=random_state, nb_classes=self.nb_classes,xnoisy_or_not=self.xnoisy_or_not)
                self.noise_targets = noisify_y(train_labels=self.targets, noise_type=ynoise_type, noise_rate=ynoise_rate, random_state=random_state, nb_classes=self.nb_classes)
                self.noise_targets = np.asarray([i[0] for i in self.noise_targets])
                self.targets =  np.asarray([i[0] for i in self.targets])
                self.ynoisy_or_not = (self.targets != self.noise_targets)
            else:
                self.noise_targets = self.targets
                self.ynoisy_or_not = np.zeros(self.data.shape[0]).astype(np.int)
            self.report_noise()

    def get_noise(self): 
        xy_noise = np.logical_and(self.xnoisy_or_not, self.ynoisy_or_not)
        x_noise = np.logical_and(self.xnoisy_or_not, ~self.ynoisy_or_not)
        y_noise = np.logical_and(~self.xnoisy_or_not, self.ynoisy_or_not)
        clean = np.logical_and(~self.xnoisy_or_not, ~self.ynoisy_or_not)
        return {
            'xy_noise': xy_noise,
            'x_noise': x_noise,
            'y_noise': y_noise,
            'clean': clean,
            'xnoisy': self.xnoisy_or_not,
            'ynoisy': self.ynoisy_or_not
        }
    def report_noise(self):
        noise_stat = self.get_noise()
        print('Noise Stat:')
        for key, val in noise_stat.items():
            print(key, val.sum())
            


    def __getitem__(self, index: int):
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
        # img = Image.fromarray(img)
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if self.train:
            return ((img, xnoisy), (noise_tar, true_tar))
        else:
            return img, target


    def __len__(self) -> int:
        return len(self.data)


class TinyImagenetNoiseMulti(VisionDataset):
    nb_classes = 200
    def __init__(
            self,
            # root: str = '/data/LargeData/Regular/tinyimagenet',
            train: bool = True,
            transform= None,
            target_transform = None,
            download: bool = False,
            xnoise_rates=[], xnoise_types=[], xnoise_args=[], 
            ynoise_type='symmetric', ynoise_rate=0,
            random_state=0
    ) -> None:

        super(TinyImagenetNoiseMulti, self).__init__(root='', transform=transform,
                                      target_transform=target_transform)
        self.types = ['clean'] + xnoise_types
        self.rates = [1-sum(xnoise_rates)] + xnoise_rates
        self.imgs = []
        self.train = train 

        if train:
            for idx, xnoise_type in enumerate(self.types):
                if xnoise_type == 'clean':
                    root = '/data/LargeData/Regular/tinyimagenet/train.npz'
                elif self.rates[idx] > 0:
                    noise_name = f'{xnoise_type}_{xnoise_args[idx-1]}'
                    root = f'/data/leyang/tiny/{noise_name}.npz'
                    if not os.path.exists(root):
                        print('Noisy image not found! Please run make_tiny_c.py first.')
                        exit(1)
                obj = np.load(root)
                self.imgs.append(obj['X'])
                self.targets = obj['Y']
        else:
            root = '/data/LargeData/Regular/tinyimagenet/val.npz'
            obj = np.load(root)
            self.imgs.append(obj['X'])
            self.targets = obj['Y']
        
        # self.data, self.targets = obj['X'], obj['Y']
        # self.data = np.uint8(self.data)

        if self.train:
            seed = np.random.RandomState(random_state)
            self.noise_select = seed.multinomial(1, self.rates, size=(self.imgs[0].shape[0],))
            self.noise_select = np.argmax(self.noise_select, axis=1)
            self.xnoisy_or_not = (self.noise_select > 0)
            if ynoise_rate > 0.0:
                self.targets = np.asarray([[int(self.targets[i])] for i in range(len(self.targets))])
                self.noise_targets = noisify_y(train_labels=self.targets, noise_type=ynoise_type, noise_rate=ynoise_rate, random_state=random_state, nb_classes=self.nb_classes)
                self.noise_targets = np.asarray([i[0] for i in self.noise_targets])
                self.targets =  np.asarray([i[0] for i in self.targets])
                self.ynoisy_or_not = (self.targets != self.noise_targets)
            else:
                self.noise_targets = self.targets
                self.ynoisy_or_not = np.zeros(self.targets.shape[0]).astype(np.int)
            self.report_noise()

    def get_noise(self): 
        xy_noise = np.logical_and(self.xnoisy_or_not, self.ynoisy_or_not)
        x_noise = np.logical_and(self.xnoisy_or_not, ~self.ynoisy_or_not)
        y_noise = np.logical_and(~self.xnoisy_or_not, self.ynoisy_or_not)
        clean = np.logical_and(~self.xnoisy_or_not, ~self.ynoisy_or_not)
        return {
            'xy_noise': xy_noise,
            'x_noise': x_noise,
            'y_noise': y_noise,
            'clean': clean,
            'xnoisy': self.xnoisy_or_not,
            'ynoisy': self.ynoisy_or_not
        }
    
    def report_noise(self):
        noise_stat = self.get_noise()
        print('Noise Stat:')
        for key, val in noise_stat.items():
            print(key, val.sum())
        print('Xnoise pattern:')
        print(self.types)
        print(self.rates)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            noise_idx = self.noise_select[index]
            img, noise_tar, true_tar, xnoisy = self.imgs[noise_idx][index], int(self.noise_targets[index]), int(self.targets[index]), self.xnoisy_or_not[index]
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
            return img, target


    def __len__(self) -> int:
        return len(self.imgs[0])

