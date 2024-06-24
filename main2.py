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
import neptune

torch.set_float32_matmul_precision('high')

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

run = neptune.init_run(
    project="fednl/cvs",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDA5OTRhYi0zYjkzLTQ4NTEtOTAwYS1hNzZhYjQ2ZjBlN2UifQ==",
)  # your credentials

class MultiLabelImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels = json.load(open(label_file, 'r'))
        self.transform = transform
        if self.transform is None:
            self.transform = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        image_name = os.path.basename(image_path)
        label = self.labels.get(image_name.split('.')[0], [])
        image = self.transform(image)
        return image, label

train_image_dir = './output_frames_train'
train_label_file = './data_train.json' 

val_image_dir = './output_frames_val'
val_label_file = './data_val.json' 

train_set = MultiLabelImageDataset(train_image_dir, train_label_file, transform=transforms)
val_set = MultiLabelImageDataset(val_image_dir, val_label_file, transform=None)

# dataset = MultiLabelImageDataset(train_image_dir, train_label_file, transform=transforms)
# train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_set, batch_size=32, shuffle=False)

print(len(train_loader))
print(len(valid_loader))

# Load the pre-trained ViT model
model = models.vit_l_16(weights='DEFAULT')
# # Modify the classification head for multi-label classification
num_classes = 3
print(model.heads.head.in_features)
model.heads = nn.Sequential(
    nn.Linear(model.heads.head.in_features, num_classes, bias=True),
)

# model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
# model.fc = nn.Sequential(nn.Linear(2048, 3, bias=True))

# ckpt = torch.load('./model_pretrained_res50.pt')

# print(ckpt)
# model = torch.load('./model_pretrained_res50.pt')
# Move the model to the configured device
# device= 'cpu'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

num_epochs = 80
best_loss=1000

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
learning_rate = 3e-5
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs*len(train_loader), eta_min=3e-6)


# Train the model

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
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    # Compute average epoch loss
    epoch_loss = running_loss / len(train_set)
    print(f'Epoch {epoch+1} Loss: {epoch_loss:.6f}')


    model.eval()
    map_score, ap_per_class, val_loss = evaluate_map(model, valid_loader, device)
    print("val_map_score", map_score)
    print("ap_per class", ap_per_class)

    run['val/mAP'].append(map_score)
    run['train/loss'].append(epoch_loss)
    run['val/loss'].append(val_loss)


    if val_loss<best_loss:
        best_loss = val_loss
        model_name = f'model_res50.pt'
        torch.save(model, model_name)
