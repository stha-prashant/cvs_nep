#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="the number of epochs: E")
    parser.add_argument('--batch_size', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--schedule', type=int, nargs='*', default=[162, 244],
                        help='Decrease learning rate at these rounds.')
    parser.add_argument('--lr_decay',type = float,default=0.1,
                        help = 'Learning rate decay at specified rounds')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--reg', default=1e-3, type=float, 
                        help='weight decay for an optimizer')
        
    # model arguments
    parser.add_argument('--model', type=str, default='resnet50', help='model name')
    parser.add_argument('--pretrained',action = 'store_true', help='whether to use pretrained models')

    parser.add_argument('--ckpt_path', type=str, default=None, help='Path of the previous checkpoint')
    parser.add_argument('--save_path', type=str, default='./save', help='Path of the previous checkpoint')
    parser.add_argument('--pkl_path', type=str, default='./', help='Path of the previous checkpoint')
    parser.add_argument('--num_frames', type=int, default = 1, help = 'Number of frames to take')
    parser.add_argument('--group', type=int, default = 0, help = 'Whether to group the frames or not')
    parser.add_argument('--dropout', type=float, default = 0.2, help = '')


    # utils arguments
    parser.add_argument('--dataset_path', type=str, default='./data', 
                        help="name of dataset")
    parser.add_argument('--gpu', default="1", 
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--seed', type=int, default=None, 
                        help='random seed')
    parser.add_argument('--neptune',action = 'store_true',
                        help='whether to use neptune')
                        
  
    
    args = parser.parse_args()
    return args
