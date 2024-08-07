import sys
import time
import json
import torch
import copy

import resource
import numpy as np

import tenseal as ts
from glob import glob
from phe import paillier
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

from models.CNNModel import LeNetMnist, LeNetCifar, AlexNetCifar, AlexNetMnist
from termcolor import colored

from SegCKKS import *

# global_pub_key, global_priv_key = paillier.generate_paillier_keypair(n_length=2048)

# HE = generate_ckks_key("32768")

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class Client():

    def __init__(self, args, dataset=None, idxs=None, w = None, FHE=None, PhePk=None, PheSk=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if args.dataset == "mnist":
            if args.model == "LeNet":
                self.model = LeNetMnist(args=args).to(args.device)
            elif args.model == "AlexNet":
                self.model = AlexNetMnist(args=args).to(args.device)
        elif args.dataset == "cifar":
            if args.model == "LeNet":
                self.model = LeNetCifar(args=args).to(args.device)
            elif args.model == "AlexNet":
                self.model = AlexNetCifar(args=args).to(args.device)
        self.model.load_state_dict(w)
        # Paillier initialization
        if self.args.mode == 'Paillier':
            self.pub_key = PhePk
            self.priv_key = PheSk
        if self.args.mode == 'CKKS':
            self.HE = FHE


    def train(self):
        start_t_train = time.time()
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)

        # client train and update
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        w_new = net.state_dict()
        print("client train time:", time.time()-start_t_train)
        update_w = {}
        comm_w = {}
        if self.args.mode == 'Plain':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
        elif self.args.mode == 'Paillier':  # Paillier Encryption
            print('paillier encrypting...')
            enc_start = time.time()
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                list_w = update_w[k].view(-1).cpu().tolist()
                for i, elem in enumerate(list_w):
                    elem = round(elem,3)
                    list_w[i] = self.pub_key.encrypt(elem)
                update_w[k] = list_w
            enc_end = time.time()
            print('Encryption time:', enc_end - enc_start)
        elif self.args.mode == 'CKKS':  # CKKS Encryption
            print('CKKS encrypting...')
            enc_start = time.time()
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                list_w = update_w[k].view(-1).cpu().tolist()
                list_w = np.array(list_w)
                vec_len = len(list_w)
                if len(list_w) <= self.HE.get_nSlots():
                    plist_w = enc_vector(self.HE, list_w)
                    update_w[k] = plist_w
                else:
                    plist_w_arr = seg_enc_vector(self.HE, list_w, vec_len)
                    update_w[k] = plist_w_arr
            enc_end = time.time()
            print('Encryption time:', colored((enc_end - enc_start), 'green'))
        else:
            raise NotImplementedError
        return update_w, sum(batch_loss) / len(batch_loss)


    def update(self, w_glob):
        print("update phrase")
        if self.args.mode == 'Plain':
            self.model.load_state_dict(w_glob) 
        elif self.args.mode == 'Paillier':  # Paillier decryption
            update_w_avg = copy.deepcopy(w_glob)
            print('paillier decrypting...')
            dec_start = time.time()
            for k in update_w_avg.keys():
                # decryption
                for i, elem in enumerate(update_w_avg[k]):
                    update_w_avg[k][i] = self.priv_key.decrypt(elem)
                # reshape to original and update
                origin_shape = list(self.model.state_dict()[k].size())
                update_w_avg[k] = torch.FloatTensor(update_w_avg[k]).to(self.args.device).view(*origin_shape)
                self.model.state_dict()[k] += update_w_avg[k]
            dec_end = time.time()
            print('Decryption time:', dec_end - dec_start)
        elif self.args.mode == 'CKKS':
            update_w_avg = copy.deepcopy(w_glob)
            print('CKKS Decrypting...')
            dec_start = time.time()
            for k in update_w_avg.keys():
                origin_shape = list(self.model.state_dict()[k].size())
                # print(np.prod(origin_shape))
                enc_vec = update_w_avg[k]
                if isinstance(enc_vec, list):
                    dec_vec = seg_dec_vector(self.HE, enc_vec)
                else:
                    dec_vec = dec_vector(self.HE, enc_vec)
                vlen = np.prod(origin_shape)
                dec_vec = dec_vec[:vlen]
                update_w_avg[k] = dec_vec
                update_w_avg[k] = torch.FloatTensor(update_w_avg[k]).to(self.args.device).view(*origin_shape)
                update_w_avg[k] = torch.div(update_w_avg[k], self.args.num_users)
                self.model.state_dict()[k] += update_w_avg[k]
            dec_end = time.time()
            print('Decryption time:', colored(dec_end - dec_start, 'green') )
        else:
            raise NotImplementedError
