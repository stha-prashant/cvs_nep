from dataset import SingleImageDataset, MultipleImageDataset
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
from options import args_parser
from models import MultiImageModel
from utils import setup_seed

if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.seed)
    if args.neptune:
        run = neptune.init_run(
            project="fednl/cvs",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NDA5OTRhYi0zYjkzLTQ4NTEtOTAwYS1hNzZhYjQ2ZjBlN2UifQ==",
        )  # your credentials
        run['parameters'] = args
    save_path = os.path.join(args.save_path, 'cvs_checkpoints')
    os.makedirs(save_path, exist_ok=True)

    ###################### Dataset AND Dataloaders ###################################
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_set = MultipleImageDataset(os.path.join(args.pkl_path, f'{args.num_frames}_images_train_{args.seed}.pkl'), args.num_frames, transform = transforms)
    val_set = MultipleImageDataset(os.path.join(args.pkl_path, f'images_val_{args.seed}.pkl'), 1, transform = transforms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    ####################################################################################

    torch.set_float32_matmul_precision('high')
    device = f'cuda:{args.gpu}'
    
    model = MultiImageModel(args)
    model = model.to(device)

    if args.ckpt_path is not None:
        raise NotImplementedError
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.reg)
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = args.epochs


    ######################### Train loop ###################################33
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        start = time.time()
        model.train()
        for step, (datas, labels) in (enumerate(train_loader)):
            datas = datas.to(device)
            labels = labels.to(device)
            output = model(datas)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # print(f'Epoch: {epoch} | Step: {step} | Time: {round((end-start)*1000)} | Loss: {loss.item()}')
        end = time.time()
        val_map, _, val_loss = evaluate_map(model, val_loader, 'cuda:0')
        print(f"Epoch {epoch} | validation mAP: {val_map} | avg_loss: {total_loss/(step+1)}")

        if args.neptune:
            run['val/mAP'].append(val_map)
            run['train/loss'].append(total_loss/(step+1))
            run['val/loss'].append(val_loss)

        torch.save({
                        'model': model.state_dict(),
                        'epoch': epoch+1,
                        'optimizer': optimizer.state_dict(),
                    }, os.path.join(save_path, f'model_n_{args.num_frames}_{args.seed}.pth'))

        if epoch+1 % 10 == 0 and args.neptune:
            run["checkpoint"].upload(os.path.join(save_path, f'model_n_{args.num_frames}_{args.seed}.pth'))
    run.stop()
