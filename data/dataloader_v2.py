import os
from PIL import Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from .augment import PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCompose
import torch
def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    train_dataset = Dataset(image_dir, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    return dataloader

def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'nature')
    test_dataset = Dataset(image_dir, is_test=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler
    )
    return dataloader

def valid_dataloader(path, batch_size=1, num_workers=0,use_transform=True):
    image_dir = os.path.join(path, 'valid')
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(128),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        Dataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader

class Dataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.image_list = os.listdir(os.path.join(image_dir, 'input/')) 
        self._check_image(self.image_list) # 檢查檔案類型
        self.label_list = list()
        for i in range(len(self.image_list)):
            filename = self.image_list[i]
            self.label_list.append(filename)
        self.image_list.sort()
        self.label_list.sort()
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'gt', self.label_list[idx]))
        image = image.convert("RGB")
        label = label.convert("RGB")
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label
    
    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError