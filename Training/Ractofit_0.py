import gc
import os.path
import torch
from Training.Trainer import Trainer
import sys
sys.path.append(os.path.dirname(
    os.path.abspath('C:/Users/Jeonghyemin/PycharmProjects/Generative_Continual_Learning/Data/data_loader.py')))
from Data import data_loader
from Data.data_loader import *
from utils import variable
import numpy as np
import math
import copy
from log_utils import save_images
import torch.optim as optim


class Ractofit_0(Trainer):
    def __init__(self, model, args):
        super(Ractofit_0, self).__init__(model, args)
        self.nb_samples_rehearsal = args.nb_samples_rehearsal
        self.data_memory = None
        self.task_samples = None
        self.task_labels = None

    
    def create_next_data(self, ind_task):
        
        # RehearsalDGZ
        if self.args.method == 'RehearsalDGZ':
            print('RehearsalDGZ')
            x_tr, y_tr = self.train_loader[ind_task].get_sample(self.nb_samples_rehearsal)
            if self.gpu_mode:
                x_tr, y_tr = x_tr.cpu(), y_tr.cpu()
            self.task_samples = x_tr.clone()
            self.task_labels = y_tr.clone()
            print(self.task_samples.shape)

            print('train_loader ', len(self.train_loader[ind_task]))
            if ind_task > 0:
                self.train_loader[ind_task].concatenate(self.data_memory)
                train_loader = self.train_loader[ind_task]
                test_loader = None
            print('train_loader ', len(self.train_loader[ind_task]))

            c1 = 0
            c2 = 1
            tasks_tr = []
            tasks_tr.append([(c1, c2), self.task_samples.clone().view(-1, 3072), self.task_labels.clone().view(-1)]) 
            if ind_task <= 0:
                self.data_memory = DataLoader(tasks_tr, self.args).increase_size(1)
            else:
                self.data_memory.concatenate(
                    DataLoader(tasks_tr, self.args).increase_size(1))
                print('data_memory ', len(self.data_memory))
        
        
        task_te_gen = None
        if ind_task > 0:
            print("Z train")
            self.model.eval() 

            hard_samples = None
            z_rands = None
            for t in range(ind_task):
                maxEpochs, batch_sz = self.args.hardsample_epoch, 5000
                lr, sz = 0.001, 1000
                batches = math.ceil(batch_sz/sz)
                for i in range(batches):
                    if self.args.dataset == 'mnist' or self.args.dataset == 'fashion':
                        Zs = Variable(torch.rand((sz, self.model.z_dim)).cuda(self.model.device), requires_grad=True)
                    elif self.args.dataset == 'cifar10':
                        Zs = Variable(torch.randn((sz, self.model.z_dim)).cuda(self.model.device), requires_grad=True)
                    z_rand = copy.deepcopy(Zs)
                    if self.args.dataset == 'cifar100':
                        y = (torch.randperm(sz * 10) % ind_task * 10)[:sz]
                    else:
                        y = (torch.randperm(sz * 10) % ind_task)[:sz]
                    y_onehot = variable(self.model.get_one_hot(y), self.args.device)
                    # print(y)

                    optZ = optim.RMSprop([Zs], lr=lr)
                    for e in range(maxEpochs):
                        optZ.zero_grad()
                        if self.args.conditional == True:
                            current_score = self.model.D(self.model.G(Zs, y_onehot), y_onehot)
                        else:
                            current_score = self.model.D(self.model.G(Zs))
                        # future_score = self.future_model.D(self.model.G(Zs))
                        loss = - (current_score)
                        loss.mean().backward()
                        optZ.step()
                    with torch.no_grad():
                        if t is 0 and i is 0:
                            if self.args.conditional == True:
                                ys = y.cpu()
                                hard_samples = self.model.G(Zs, y_onehot).cpu()
                                z_rands = self.model.G(z_rand, y_onehot).cpu()
                            else:
                                hard_samples = self.model.G(Zs).cpu()
                                z_rands = self.model.G(z_rand).cpu()
                        else:
                            if self.args.conditional == True:
                                ys = torch.cat((ys, y.cpu()))
                                hard_samples = torch.cat((hard_samples, self.model.G(Zs, y_onehot).cpu()))
                                z_rands = torch.cat((z_rands, self.model.G(z_rand, y_onehot).cpu()))
                            else:
                                hard_samples = torch.cat((hard_samples, self.model.G(Zs).cpu()))
                                z_rands = torch.cat((z_rands, self.model.G(z_rand).cpu()))
                    print(hard_samples.shape)


#             save_images(hard_samples.data.numpy().transpose(0, 2, 3, 1)[:20 * 20, :, :, :], [20, 20],
#                         './forgettable_z/CL_' + self.args.dataset + '_iter_' + str(maxEpochs) + '_task_' + str(ind_task) + '.png')
#             save_images(z_rands.data.numpy().transpose(0, 2, 3, 1)[:20 * 20, :, :, :], [20, 20],
#                         './forgettable_z/CL_' + self.args.dataset + '_iter_' + str(maxEpochs) + '_task_' + str(ind_task) + '_rand.png')



            # 3. add hardsamples to train_loader
            print("3. add hardsamples to train loader")
            print('before train_loader: ', len(self.train_loader[ind_task]))
            self.train_loader[ind_task]
            if self.args.conditional == False:
                ys = torch.ones(hard_samples.shape[0], 1)  # 걍 1로 줘버림
            hard_samples_loader = data_loader.DataLoader([[(0, 1), hard_samples.view(-1,
                            self.input_size * self.image_size * self.image_size), ys.view(-1)]], self.args)
            self.train_loader.concatenate(hard_samples_loader)


            train_loader = self.train_loader[ind_task]
            print('after train_loader: ', len(self.train_loader[ind_task]))
#             my_visualize_sample(train_loader, ind_task, self.args)
            train_loader.shuffle_task()


            if task_te_gen is not None:
                self.test_loader.concatenate(task_te_gen)
                test_loader = self.test_loader[ind_task]
                test_loader.shuffle_task()
            else:
                test_loader = None
        else:
            train_loader = self.train_loader[ind_task]
            test_loader = self.test_loader[ind_task]
        return train_loader, test_loader
