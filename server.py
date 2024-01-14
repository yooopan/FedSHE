import time
import torch
import copy
import numpy as np
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.CNNModel import LeNetMnist, LeNetCifar, AlexNetCifar,AlexNetMnist

class Server():

    def __init__(self, args, w):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        if args.dataset == "mnist":
            if args.model == "LeNet":
                self.model = LeNetMnist(args=args).to(args.device)
            elif args.model == "AlexNet":
                self.model = AlexNetMnist(args=args).to(args.device)
            elif args.model == "CNN":
                self.model = CNNMnist(args=args).to(args.device)
        elif args.dataset == "cifar":
            if args.model == "LeNet":
                self.model = LeNetCifar(args=args).to(args.device)
            elif args.model == "AlexNet":
                self.model = AlexNetCifar(args=args).to(args.device)
        self.model.load_state_dict(w)


    def FedAvg(self):
        print("server fedavg phrase")
        start_avg_time = time.time()
        if self.args.mode == 'Plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                self.model.state_dict()[k] += update_w_avg[k]  
            print("agg time:", time.time() - start_avg_time) 
            return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)
        elif self.args.mode == 'Paillier':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                client_num = len(self.clients_update_w)
                for i in range(1, client_num):
                    for j in range(len(update_w_avg[k])):  # element-wise sum
                        update_w_avg[k][j] += self.clients_update_w[i][k][j]
                for j in range(len(update_w_avg[k])):  # element-wise avg
                    update_w_avg[k][j] /= client_num
            print("agg time:", time.time() - start_avg_time)
            return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)
        elif self.args.mode == 'CKKS':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                client_num = len(self.clients_update_w)
                for i in range(1, client_num):
                    enc_k_value = self.clients_update_w[i][k]
                    if isinstance(enc_k_value, list):
                        for j in range(len(enc_k_value)):
                            update_w_avg[k][j] += enc_k_value[j]
                    else:
                        update_w_avg[k] += self.clients_update_w[i][k]
            print("agg time:", time.time() - start_avg_time)
            return update_w_avg, sum(self.clients_loss) / len(self.clients_loss)


    def mnist_test(self, datatest):
        self.model.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy.item(), test_loss


    def cifar_test(self, dataset):
        self.model.eval()
        testloader = DataLoader(dataset, batch_size=self.args.bs)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        for images, labels in testloader:
            output = self.model(images)
            loss = criterion(output, labels)
            test_loss += loss.item()*images.size(0)
            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            for i in range(len(labels)):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] +=1
        test_loss = test_loss/len(testloader.sampler)
        print('Test Loss: {:.6f}\n'.format(test_loss))
        for i in range(10):
          if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%
                  (str(i), 100 * class_correct[i]/class_total[i],
                   np.sum(class_correct[i]), np.sum(class_total[i])))
          else:
            print('Test Accuracy of %5s: N/A(no training examples)' % classes[i])
        accuracy = 100.00 * np.sum(class_correct)  / np.sum(class_total)
        print('\nTest Accuracy (Overall): %.4f%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        return accuracy, test_loss
