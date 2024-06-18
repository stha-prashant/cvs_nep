from dataset import SingleImageDataset
from evaluate import evaluate_map
import torchvision
import torch
from torch.optim import AdamW
import torch.nn as nn
from torchvision.transforms import v2
import time
from torch.utils.data import DataLoader
import neptune
import os
from tqdm import tqdm

run = neptune.init_run(
    project="fednl/cvs",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDA5OTRhYi0zYjkzLTQ4NTEtOTAwYS1hNzZhYjQ2ZjBlN2UifQ==",
)  # your credentials
save_path = '/raid/binod/prashant/cvs_checkpoints'
os.makedirs(save_path, exist_ok=True)

###################### Dataset AND Dataloaders ###################################
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
all_data = SingleImageDataset('single_images.pkl', transform=transforms)
train_set, val_set = torch.utils.data.random_split(all_data, [0.8, 0.2])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
####################################################################################

torch.set_float32_matmul_precision('high')
device = 'cuda:0'
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Sequential(nn.Linear(2048, 3, bias=True))
model = model.to(device)
# model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()
epochs = 50


######################### Train loop ###################################33
for epoch in tqdm(range(epochs)):
    total_loss = 0
    start = time.time()

    for step, (datas, labels) in (enumerate(train_loader)):
        datas = datas.to(device)
        labels = labels.to(device)
        output = model(datas)

        loss = criterion(output, labels)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        # print(f'Epoch: {epoch} | Step: {step} | Time: {round((end-start)*1000)} | Loss: {loss.item()}')
    end = time.time()
    val_map, _, val_loss = evaluate_map(model, val_loader, 'cuda:0')
    print(f"Epoch {epoch} | validation mAP: {val_map} | avg_loss: {total_loss/(step+1)}")

    run['val/mAP'].append(val_map)
    run['train/loss'].append(total_loss/(step+1))
    run['val/loss'].append(val_loss)

    torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch+1,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(save_path, 'model.pkl'))

    if epoch+1 % 10 == 0:
        run["checkpoint"].upload(os.path.join(save_path, 'model.pkl'))
run.stop()
