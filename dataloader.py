import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets
import os
PBN_DATAPATH = './data'

def load_cifar10():
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = torchvision.datasets.CIFAR10(root=PBN_DATAPATH, 
                                            train=True,
                                            download=True, 
                                            transform=train_transform)

    test_data = torchvision.datasets.CIFAR10(root=PBN_DATAPATH, 
                                            train=False,
                                            download=True, 
                                            transform=test_transform)
    return train_data, test_data


def load_cifar100():
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
        
    train_data = torchvision.datasets.CIFAR100(root=PBN_DATAPATH, 
                                            train=True,
                                            download=True, 
                                            transform=train_transform)

    test_data = torchvision.datasets.CIFAR100(root=PBN_DATAPATH, 
                                            train=False,
                                            download=True, 
                                            transform=test_transform)
    return train_data, test_data


from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset



class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)


    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, (label-1)


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)


def load_flower102():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), 
    #     transforms.Resize(size=(224, 224), antialias=True),
    #     transforms.Normalize([0.4330, 0.3819, 0.2964], [0.2945, 0.2465, 0.2735]), 
    # ])
    mean = [0.4330, 0.3819, 0.2964]
    std = [0.2945, 0.2465, 0.2735]
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)])
    

    train_data = Flowers102(root=PBN_DATAPATH, 
                            split='train',
                            download=True, 
                            transform=train_transform)

    test_data = Flowers102(root=PBN_DATAPATH,  
                            split='test',
                            download=True, 
                            transform=test_transform)
    return train_data, test_data


def load_tiny_imagenet_200():
    mean = [x / 255 for x in [127.5, 127.5, 127.5]]
    std = [x / 255 for x in [127.5, 127.5, 127.5]]
    # Puzzle Mix
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    train_root = os.path.join('./data/tiny-imagenet-200/train')  # this is path to training images folder
    validation_root = os.path.join('./data/tiny-imagenet-200/val/images')  # this is path to validation images folder
    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(validation_root, transform=test_transform)
    
    return train_data, test_data


def load_data(batch_size, data_name):

    if data_name.lower() == 'cifar10':
        train_data, test_data = load_cifar10()
    elif data_name.lower() == 'cifar100':
        train_data, test_data = load_cifar100()
    elif data_name.lower() == 'flower102':
        train_data, test_data = load_flower102()
    elif data_name.lower() == 'tinyimagenet200':
        train_data, test_data = load_tiny_imagenet_200()
    else:
        raise f'{data_name} is not supported'
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    dataloaders = {
        "train": trainloader,
        "val": testloader
    }

    return dataloaders