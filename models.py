import torchvision 
import torch.nn as nn
import torch
import torch.nn.functional as F
import timm

class ResNetBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model == 'resnet50':
            weights = 'DEFAULT' if args.pretrained else None
            self.feature_extractor = torchvision.models.resnet50(weights=weights)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

    def forward(self, x):
        batch_size, *_ = x.shape
        representations = self.feature_extractor(x).view(batch_size, -1)
        return representations


class MultiImageModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if args.model == 'resnet50':
            weights = 'DEFAULT' if args.pretrained else None
            self.feature_extractor = torchvision.models.resnet50(weights=weights)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            # for param in self.feature_extractor.parameters():
            #     param.requires_grad = False
            embedd_dim = self.args.num_frames * 2048 if args.group else 2048
            self.classifier = nn.Sequential(
                nn.Linear(embedd_dim, 512),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(512, 3)
            )
        elif args.model == 'vit':
            self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=args.pretrained)
            self.feature_extractor.head = nn.Identity()
            embedd_dim = self.args.num_frames * self.feature_extractor.embed_dim if args.group else self.feature_extractor.embed_dim
            self.classifier = nn.Sequential(
                nn.Linear(embedd_dim, 512),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(512, 3),
            )

    def forward(self, x):
        if self.args.num_frames > 1 and self.args.group == 1:
            # take multiple images
            batch_size, n_frames, n, c, w = x.shape
            x = x.view(batch_size * n_frames, n, c, w)
            representations = self.feature_extractor(x)
            representations = representations.reshape(batch_size, -1)
            z = self.classifier(representations)
            # representations = representations.view(batch_size, n_frames, -1)
            # representations = torch.sum(representations * self.weights.unsqueeze(-1), dim = 1)
            return z, None
        else:
            batch_size, *_ = x.shape
            representations = self.feature_extractor(x).view(batch_size, -1)
            logits = self.classifier(representations)

        # logits = self.fc(z)
        return logits, None

    def infer(self, x):
        batch_size, n_frames, n, c, w = x.shape
        assert n_frames == 1, "Number of frames for inference should be 1"
        x = x.view(batch_size * n_frames, n, c, w)
        z = self.feature_extractor(x)
        z = z.view(batch_size, n_frames, -1)
        representations = z.copy()
        z =  z.view(batch_size, -1)

        logits = self.fc(z)
        return logits, representations