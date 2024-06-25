import torch
import numpy as numpy
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import pickle
from PIL import Image


class SingleImageDataset(Dataset):
    def __init__(self, pkl_file, transform=None):

        self.transform = transform
        if transform is None:
            self.transform = v2.Compose([
                v2.Resize((224,224)),
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
        # if self.transform:
        image = self.transform(image)
        label = torch.tensor(item['label'], dtype=torch.float32)
        return image, label

class MultipleImageDataset(Dataset):
    def __init__(self, pkl_file, args, transform=None):
        self.transform = transform
        if transform is None:
            self.transform = v2.Compose([
                v2.Resize(224),
                v2.ToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.num_frames = args.num_frames if args.group else 1
        self.args = args
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        images = item['img_array']
        if self.args.group:
            transformed_images = [self.transform(Image.fromarray(img)) for img in images]

            assert len(transformed_images) <= self.num_frames
            if len(transformed_images) < self.num_frames:
                while len(transformed_images) < self.num_frames:
                    transformed_images.extend(transformed_images[:self.num_frames - len(transformed_images)])

            images_tensor = torch.stack(transformed_images)
        else:
            image = Image.fromarray(item['img_array'])
            if self.transform:
                images_tensor = self.transform(image)
        label = torch.tensor(item['label'], dtype=torch.float32)
        return images_tensor, label


