import torch
import numpy as numpy
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import pickle
from PIL import Image


class SingleImageDataset(Dataset):
    def __init__(self, pkl_file, transform=None):

        self.transform = transform
        if self.transform is None:
            transforms = v2.Compose([
                v2.Resize(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.fromarray(item['img_array'])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(item['label'], dtype=torch.float32)
        return image, label


