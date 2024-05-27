from typing import Any
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image , target) :
        for step in self.transforms:
            image, target = step(image, target)
        return image, target

class Resize(object):
    def __init__(self, output_size=224) -> None:
        self.size = output_size
        # self.train = train
    def __call__(self, image, target) -> Any:
        image = F.resize(image, self.size)

        # if self.train:
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        
        return image, target

class Totensor(object):
    def __call__(self, image, target) :
        image = F.to_tensor(image)
        target = torch.tensor(np.asarray(target), dtype=torch.int64)
        return image, target

class ColorJitter(object):
    def __init__(self) -> None:
        self.aug = T.ColorJitter(0.1, 0.1, 0.1, 0.1)
        self.p = 0.2

    def __call__(self, image, target) :
        a = torch.rand(1).item()
        if a < self.p:
            image = self.aug(image)
        return image, target

class Normalize(object):
    def __init__(self, mean, std) :
        self.mean = mean
        self.std = std
    
    def __call__(self, image , target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class ToNumpy(object):
    def __call__(self, image, target):
        image = np.array(image)
        target = np.array(target)
        return image, target

class ToPILImage(object):
    def __call__(self, image, target):
        image = Image.fromarray(image)
        target = Image.fromarray(target)
        return image, target
    
def get_transform(size, train=True):
    transforms = []
        
    transforms.append(Resize(size))
    transforms.append(Totensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return Compose(transforms)


