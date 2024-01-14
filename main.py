import faulthandler
faulthandler.enable()
from statistics import mode
import numpy as np
import time
import torch
from torchvision import datasets, transforms, utils
from models.CNNModel import LeNetMnist, LeNetCifar, AlexNetCifar, AlexNetMnist

from options import args_parser
from client import *
from server import *
import copy
import resource
from termcolor import colored
import matplotlib.pyplot as plt

from phe import paillier
from SegCKKS import *

def load_dataset(dataset="mnist"):
    if dataset == "mnist":
        trans_mnist = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        dataset_train = datasets.MNIST(
            root='./data/mnist/', train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.MNIST(
            root='./data/mnist/', train=False, download=True, transform=trans_mnist
        )
        return dataset_train, dataset_test
    elif dataset == "cifar":
        trans_cifar = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        dataset_train = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=trans_cifar
        )
        dataset_test = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=trans_cifar
        )
        return dataset_train, dataset_test


def create_client_server():
    if args.mode  == "CKKS":
        HE = generate_ckks_key(args.ckks_sec_level, args.ckks_mul_depth, args.ckks_key_len)
    elif args.mode == "Paillier":
        phe_pk, phe_sk = paillier.generate_paillier_keypair(n_length=args.phe_key_len)

    num_items = int(len(dataset_train) / args.num_users)
    clients, all_idxs = [], [i for i in range(len(dataset_train))]

    if args.dataset == "mnist":
        if args.model == "LeNet":
            net_glob = LeNetMnist(args=args).to(args.device)
        elif args.model == "AlexNet":
            net_glob = AlexNetMnist(args=args).to(args.device)
        elif args.model == "CNN":
            net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == "cifar":
        if args.model == "LeNet":
            net_glob = LeNetCifar(args=args).to(args.device)
        elif args.model == "AlexNet":
            net_glob = AlexNetCifar(args=args).to(args.device)

    for i in range(args.num_users):
        new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - new_idxs)
        if args.mode == "Plain":
            new_client = Client(args=args, dataset=dataset_train, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()))
        if args.mode == "CKKS":
            new_client = Client(args=args, dataset=dataset_train, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()), FHE=HE)
        elif args.mode == "Paillier":
            new_client = Client(args=args, dataset=dataset_train, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()), PhePk=phe_pk, PheSk=phe_sk)
        clients.append(new_client)
        

    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()))
    return clients, server


if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu !=-1  else 'cpu')

    args.num_channels = 1 if args.dataset == "mnist" else 3
    print(args.device)
    print("args: ", args)
    print("load dataset...")
    if args.dataset == "mnist":
        dataset_train, dataset_test = load_dataset("mnist")
    elif args.dataset == "cifar":
        dataset_train, dataset_test = load_dataset("cifar")
    print("clients and server initialization...")
    clients, server = create_client_server()


    # statistics for plot
    all_acc_train = []
    all_acc_test = []
    all_loss_glob = []
    all_loss_test = []

    train_time_epochs = []

    # training
    print("start training...")
    print('Homomorphic Encryption Scheme:', colored(args.mode, 'red'))
    start_time = time.time()
    train_time_epochs.append(start_time)
    print("start time(): ", colored(start_time, 'green'))
    for iter in range(args.epochs):
        epoch_start = time.time()
        server.clients_update_w, server.clients_loss = [], []
        for idx in range(args.num_users):
            update_w, loss = clients[idx].train()
            server.clients_update_w.append(update_w)
            server.clients_loss.append(loss)
        # calculate global weights
        w_glob, loss_glob = server.FedAvg()
        # update local weights
        for idx in range(args.num_users):
            clients[idx].update(w_glob)

        epoch_end = time.time()
        train_time_epochs.append(epoch_end)
        print(colored('=====Epoch {:3d}====='.format(iter), 'yellow'))
        print('Training time:', epoch_end, epoch_start)

        if (args.mode == 'Paillier') or ( args.mode == 'CKKS'):
            server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))

        if args.dataset == "mnist":
            acc_train, loss_train = server.mnist_test(dataset_train)
            acc_test, loss_test = server.mnist_test(dataset_test)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print('Server average loss: {:.3f}'.format(loss_glob))
            print('Test loss: {:.3f}'.format(loss_test))
            all_acc_train.append(acc_train)
            all_acc_test.append(acc_test)
            all_loss_glob.append(loss_glob)
            all_loss_test.append(loss_test)
        elif args.dataset == "cifar":
            acc_train, loss_train = server.cifar_test(dataset_train)
            acc_test, loss_test = server.cifar_test(dataset_test)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))
            print('Server average loss: {:.3f}'.format(loss_glob))
            print('Test loss: {:.3f}'.format(loss_test))
            all_acc_train.append(acc_train)
            all_acc_test.append(acc_test)
            all_loss_glob.append(loss_glob)
            all_loss_test.append(loss_test)

    print("all took time(): ", time.time() - start_time)
    # plot learning curve
    print("acc_train:", all_acc_train)
    print("acc_test:", all_acc_test)
    print("loss_glob:", all_loss_glob)
    print("loss_test:", all_loss_test)
    print("train_time_epochs：", train_time_epochs)
