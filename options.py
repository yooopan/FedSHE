#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # experiment arguments
    parser.add_argument('--mode', type=str, default='plain', help="plain, DP, or Paillier")
    parser.add_argument('--phe_key_len', type=int, default=2048, help="128,256,512,1024,2048")

    parser.add_argument('--ckks_sec_level', type=str, default='128', help="128,192,256")
    parser.add_argument('--ckks_mul_depth', type=str, default='0', help="0,1,2,3,4")
    parser.add_argument('--ckks_key_len', type=str, default='1024', help="1024,2048,3072,4096,8192,16384,32768")
    parser.add_argument('--dataset', type=str, default='mnist', help="mnist, cifar")
    parser.add_argument('--model', type=str, default='LeNet', help="LeNet, AlexNet")

    # federated arguments
    parser.add_argument('--epochs', type=int, default=30, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.015, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")


    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--no-plot', action="store_true", default=False, help="plot learning curve")
    args = parser.parse_args()
    return args
