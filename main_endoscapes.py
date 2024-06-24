import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import tqdm
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from evaluate import evaluate_map
import pandas as pd

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MultiLabelImageDataset(Dataset):
    def __init__(self, image_dir, label_df, transform=None):
        all_images_path = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = label_df
        self.image_paths = [i for i in all_images_path if i.split('/')[-1] in self.labels['file_name'].values]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        image_name = os.path.basename(image_path)
        label = self.labels.loc[self.labels['file_name'] == image_name, 'ds']
        label = label.values[0]
        label = [int(i>0.5) for i in label]

        # print(image_name, label)
        if self.transform:
            image = self.transform(image)

        return image, label

image_dir = '/raid/binod/prashant/endoscapes/all'
label_file = '/raid/binod/prashant/endoscapes/annotation_ds_coco.json' 

label_json = json.load(open(label_file, 'r'))

label_df = pd.DataFrame(label_json['images'])
print(label_df.shape)

dataset = MultiLabelImageDataset(image_dir, label_df, transform=transforms)
train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_set, batch_size=32, shuffle=False)

print(len(train_loader))
print(len(valid_loader))


# Load the pre-trained ViT model
# model = models.vit_l_16(weights='DEFAULT')
# Modify the classification head for multi-label classification
# num_classes = 3
# print(model.heads.head.in_features)
# model.heads = nn.Sequential(
#     nn.Linear(model.heads.head.in_features, num_classes, bias=True),
# )
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Sequential(nn.Linear(2048, 3, bias=True))

# Move the model to the configured device
# device= 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
learning_rate = 1e-4
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
num_epochs = 80
best_loss = 1000
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for inputs, labels in tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        inputs = inputs.to(device)
        # print(len(inputs), labels)
        labels = torch.stack(labels, dim=1)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        # outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels.float())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * inputs.size(0)

    # Compute average epoch loss
    epoch_loss = running_loss / len(train_set)
    print(f'Epoch {epoch+1} Loss: {epoch_loss:.6f}')


    model.eval()
    map_score, ap_per_class, loss = evaluate_map(model, valid_loader, device)
    print("val_map_score", map_score)
    print("ap_per class", ap_per_class)

    if loss<best_loss:
        model_name = f'model_pretrain.pt'
        torch.save(model, model_name)