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
import torch.nn.functional as F

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
        v2.Resize(size=(224, 224), antialias=True),
        # v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # all_data = SingleImageDataset('single_images.pkl', transform=transforms)
    if args.num_frames == 1 or args.group == 0:
        train_set = SingleImageDataset(os.path.join(args.pkl_path, f'{args.num_frames}_images_train_group_{args.group}_{args.seed}.pkl'), transform=transforms)
        val_set = SingleImageDataset(os.path.join(args.pkl_path, f'images_val_{args.seed}.pkl'), transform=None)
        breakpoint()
    else:
        print('Taking multiple frames')
        # assert args.num_frames >= 2, "Number of frames cannot be less than one"
        num_frames = args.num_frames if args.group else 1
        train_set = MultipleImageDataset(os.path.join(args.pkl_path, f'{args.num_frames}_images_train_group_{args.group}_{args.seed}.pkl'), args, transform = transforms)
        val_set = MultipleImageDataset(os.path.join(args.pkl_path, f'images_val_{args.seed}.pkl'), args, transform=transforms)

    args.num_frames = args.num_frames if args.group else 1
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    ####################################################################################

    # torch.set_float32_matmul_precision('high')
    device = f'cuda:{args.gpu}'
    
    # if args.num_frames == 1:
    #     if args.model == 'resnet50':
    #         weights = 'DEFAULT' if args.pretrained else None
    #         model = torchvision.models.resnet50(weights=weights)
    #         model.fc = nn.Sequential(nn.Linear(2048, 3, bias=True))
    # else:
    #     model = MultiImageModel(args)
    model = MultiImageModel(args)
    model = model.to(device)

    if args.ckpt_path is not None:
        raise NotImplementedError
    # model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    epochs = args.epochs


    ######################### Train loop ###################################33
    # evaluate_map(model, val_loader, 'cuda:0')
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        start = time.time()
        model.train()
        for step, (datas, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            datas = datas.to(device)
            labels = labels.to(device)
            output, _ = model(datas)

            loss = F.binary_cross_entropy_with_logits(output, labels)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            # print(f'Epoch: {epoch} | Step: {step} | Time: {round((end-start)*1000)} | Loss: {loss.item()}')
        end = time.time()
        val_map, _, val_loss = evaluate_map(model, val_loader, 'cuda:0')
        print(f"Epoch {epoch} | validation mAP: {val_map} | avg_loss: {total_loss/(step+1)}")

        torch.cuda.empty_cache()
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
