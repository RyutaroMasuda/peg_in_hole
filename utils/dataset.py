#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import v2 as transforms
except ImportError:
    from torchvision import transforms


class ImgDataset(Dataset):
    """
    This class is used to train models that deal only with imgs, such as autoencoders.
    Data augmentation is applied to the given image data by adding lightning, contrast, horizontal and vertical shift, and gaussian noise.

    Arguments:
        data (numpy.array): Set the data type (train/test). If the last three dimensions are HWC or CHW, `data` allows any number of dimensions.
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, data, device="cpu", stdev=None):
        """
        Reshapes and transforms the data.

        Arguments:
            data (numpy.array): The imgs data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            stdev (float, optional): The standard deviation for the normal distribution to generate gaussian noise.
        """

        self.stdev = stdev
        self.device = device
        _image_flatten = data.reshape(((-1,) + data.shape[-3:]))
        self.image_flatten = torch.Tensor(_image_flatten).to(self.device)

        self.transform_affine = transforms.Compose(
            [
                transforms.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15)),
                transforms.RandomAutocontrast(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        ).to(self.device)

        self.transform_noise = transforms.Compose(
            [
                transforms.ColorJitter(
                    contrast=[0.6, 1.4], brightness=0.4, saturation=[0.6, 1.4], hue=0.04
                )
            ]
        ).to(self.device)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            length (int): The length of the dataset.
        """

        return len(self.image_flatten)

    def __getitem__(self, idx):
        """
        Extracts a single image from the dataset and returns two imgs: the original image and the image with noise added.

        Args:
            idx (int): The index of the element.

        Returns:
            image_list (list): A list containing the transformed and noise added image (x_img) and the affine transformed image (y_img).
        """
        img = self.image_flatten[idx]

        if self.stdev is not None:
            y_img = self.transform_affine(img)
            x_img = self.transform_noise(y_img) + torch.normal(
                mean=0, std=self.stdev, size=y_img.shape, device=self.device
            )
        else:
            y_img = img
            x_img = img

        return [x_img, y_img]


class Img1VecDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., imgs, vecs), such as CNNRNN/SARNN.

    Args:
        imgs (numpy array): Set of imgs in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        vecs (numpy array): Set of vecs in the dataset, expected to be a 3D array [data_num, seq_num, state_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, imgs, vecs, device="cpu", stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the imgs, vecs, and transformation.

        Args:
            imgs (numpy array): The imgs data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            vecs (numpy array): The vecs data, expected to be a 3D array [data_num, seq_num, state_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.stdev = stdev
        self.device = device
        self.imgs = torch.Tensor(imgs).to(self.device)
        self.vecs = torch.Tensor(vecs).to(self.device)
        self.transform = nn.Sequential(
            transforms.RandomErasing(),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(self.device)

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of imgs and vecs at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and state (x_img, x_vec) and the original image and state (y_img, y_vec).
        """
        x_img = self.imgs[idx]
        x_vec = self.vecs[idx]
        y_img = self.imgs[idx]
        y_vec = self.vecs[idx]

        if self.stdev is not None:
            x_img = self.transform(y_img) + torch.normal(
                mean=0, std=0.02, size=x_img.shape, device=self.device
            )
            x_vec = y_vec + torch.normal(
                mean=0, std=self.stdev, size=y_vec.shape, device=self.device
            )

        return [[x_img, x_vec], [y_img, y_vec]]
    

class Img4VecDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., imgs, vecs), such as CNNRNN/SARNN.

    Args:
        imgs (numpy array): Set of imgs in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        vecs (numpy array): Set of vecs in the dataset, expected to be a 3D array [data_num, seq_num, state_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, imgs1, imgs2, imgs3, imgs4, vecs, device="cpu", stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the imgs, vecs, and transformation.

        Args:
            imgs (numpy array): The imgs data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            vecs (numpy array): The vecs data, expected to be a 3D array [data_num, seq_num, state_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        # import ipdb; ipdb.set_trace()
        
        self.stdev = stdev
        self.device = device
        self.imgs1 = torch.Tensor(imgs1).to(self.device)
        self.imgs2 = torch.Tensor(imgs2).to(self.device)
        self.imgs3 = torch.Tensor(imgs3).to(self.device)
        self.imgs4 = torch.Tensor(imgs4).to(self.device)

        
        self.vecs = torch.Tensor(vecs).to(self.device)
        self.transform = nn.Sequential(
            transforms.RandomErasing(),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(self.device)

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.imgs1)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of imgs and vecs at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and state (x_img, x_vec) and the original image and state (y_img, y_vec).
        """
        x_img1 = self.imgs1[idx]
        x_img2 = self.imgs2[idx]
        x_img3 = self.imgs3[idx]
        x_img4 = self.imgs4[idx]
        x_vec = self.vecs[idx]
        
        y_img1 = self.imgs1[idx]
        y_img2 = self.imgs2[idx]
        y_img3 = self.imgs3[idx]
        y_img4 = self.imgs4[idx]
        y_vec = self.vecs[idx]

        if self.stdev is not None:
            x_img1 = self.transform(y_img1) + torch.normal(
                mean=0, std=0.02, size=x_img1.shape, device=self.device
            )
            x_img2 = self.transform(y_img2) + torch.normal(
                mean=0, std=0.02, size=x_img2.shape, device=self.device
            )
            x_img3 = self.transform(y_img3) + torch.normal(
                mean=0, std=0.02, size=x_img3.shape, device=self.device
            )
            x_img4 = self.transform(y_img3) + torch.normal(
                mean=0, std=0.02, size=x_img4.shape, device=self.device
            )
            
            x_vec = y_vec + torch.normal(
                mean=0, std=self.stdev, size=y_vec.shape, device=self.device
            )

        return [[x_img1, x_img2, x_img3, x_img4, x_vec], [y_img1, y_img2, y_img3, y_img4, y_vec]]



class Img3VecDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., imgs, vecs), such as CNNRNN/SARNN.

    Args:
        imgs (numpy array): Set of imgs in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        vecs (numpy array): Set of vecs in the dataset, expected to be a 3D array [data_num, seq_num, state_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, imgs1, imgs2, imgs3, vecs, device="cpu", stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the imgs, vecs, and transformation.

        Args:
            imgs (numpy array): The imgs data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            vecs (numpy array): The vecs data, expected to be a 3D array [data_num, seq_num, state_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        # import ipdb; ipdb.set_trace()
        """
        (12, 310, 3, 64, 128)
        ipdb> imgs2.shape
        (12, 310, 3, 64, 128)
        ipdb> imgs3.shape
        (12, 310, 3, 32, 64)
        ipdb> vecs.shape
        (12, 310, 18)
        """
        """
        ipdb> imgs1.shape
        (3, 310, 3, 64, 128)
        ipdb> imgs2.shape
        (3, 310, 3, 64, 128)
        ipdb> imgs3.shape
        (3, 310, 3, 32, 64)
        ipdb> vecs.shape
        (3, 310, 18)
        """
        self.stdev = stdev
        self.device = device
        self.imgs1 = torch.Tensor(imgs1).to(self.device)
        self.imgs2 = torch.Tensor(imgs2).to(self.device)
        self.imgs3 = torch.Tensor(imgs3).to(self.device)
        
        self.vecs = torch.Tensor(vecs).to(self.device)
        self.transform = nn.Sequential(
            transforms.RandomErasing(),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(self.device)

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.imgs1)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of imgs and vecs at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and state (x_img, x_vec) and the original image and state (y_img, y_vec).
        """
        x_img1 = self.imgs1[idx]
        x_img2 = self.imgs2[idx]
        x_img3 = self.imgs3[idx]
        x_vec = self.vecs[idx]
        
        y_img1 = self.imgs1[idx]
        y_img2 = self.imgs2[idx]
        y_img3 = self.imgs3[idx]
        y_vec = self.vecs[idx]

        if self.stdev is not None:
            x_img1 = self.transform(y_img1) + torch.normal(
                mean=0, std=0.02, size=x_img1.shape, device=self.device
            )
            x_img2 = self.transform(y_img2) + torch.normal(
                mean=0, std=0.02, size=x_img2.shape, device=self.device
            )
            x_img3 = self.transform(y_img3) + torch.normal(
                mean=0, std=0.02, size=x_img3.shape, device=self.device
            )
            
            x_vec = y_vec + torch.normal(
                mean=0, std=self.stdev, size=y_vec.shape, device=self.device
            )

        return [[x_img1, x_img2, x_img3, x_vec], [y_img1, y_img2, y_img3, y_vec]]




class Img2VecDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., imgs, vecs), such as CNNRNN/SARNN.

    Args:
        imgs (numpy array): Set of imgs in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        vecs (numpy array): Set of vecs in the dataset, expected to be a 3D array [data_num, seq_num, state_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, imgs1, imgs2, vecs, device="cpu", stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the imgs, vecs, and transformation.

        Args:
            imgs (numpy array): The imgs data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            vecs (numpy array): The vecs data, expected to be a 3D array [data_num, seq_num, state_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.stdev = stdev
        self.device = device
        self.imgs1 = torch.Tensor(imgs1).to(self.device)
        self.imgs2 = torch.Tensor(imgs2).to(self.device)
        
        self.vecs = torch.Tensor(vecs).to(self.device)
        self.transform = nn.Sequential(
            transforms.RandomErasing(),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=[0.6, 1.4]),
            transforms.ColorJitter(hue=[0.0, 0.04]),
            transforms.ColorJitter(saturation=[0.6, 1.4]),
        ).to(self.device)

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.imgs1)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of imgs and vecs at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and state (x_img, x_vec) and the original image and state (y_img, y_vec).
        """
        x_img1 = self.imgs1[idx]
        x_img2 = self.imgs2[idx]
        x_vec = self.vecs[idx]
        
        y_img1 = self.imgs1[idx]
        y_img2 = self.imgs2[idx]
        y_vec = self.vecs[idx]

        if self.stdev is not None:
            x_img1 = self.transform(y_img1) + torch.normal(
                mean=0, std=0.02, size=x_img1.shape, device=self.device
            )
            x_img2 = self.transform(y_img2) + torch.normal(
                mean=0, std=0.02, size=x_img2.shape, device=self.device
            )
            x_vec = y_vec + torch.normal(
                mean=0, std=self.stdev, size=y_vec.shape, device=self.device
            )

        return [[x_img1, x_img2, x_vec], [y_img1, y_img2, y_vec]]







class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return (
            len(self.sampler)
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
