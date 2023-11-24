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


class Ractofit(Trainer):
    def __init__(self, model, args):
        super(Ractofit, self).__init__(model, args)
        self.x_hats = torch.Tensor()
        self.y_hats = torch.Tensor()

    def create_next_data(self, ind_task):
        task_te_gen = None
        new_train_loader = None

        if ind_task > 0:
            print('<<prepare dataset for new task train>>')
            print('<step 1> add rehearsal data')
            self.train_loader[ind_task]
            train_loader = self.train_loader[ind_task]

           
                
            with torch.no_grad():
                self.model.G.eval()
                if self.args.conditional:
                    y_hat = torch.tensor([i // self.args.num_z for i in range(ind_task * self.args.num_z)])
                    y_hat_onehot = variable(self.model.get_one_hot(y_hat), self.args.device)
                    x_hat = self.previous_model.G(variable(torch.Tensor(self.z_rehearsal), self.args.device), y_hat_onehot)
                else:
                    x_hat = self.previous_model.G(variable(torch.Tensor(self.z_rehearsal), self.args.device))
                    self.model.expert.net.eval()
                    y_hat = self.model.expert.labelize(x_hat, ind_task)
                x_hat = x_hat.cpu()
                y_hat = y_hat.cpu()
                x_hat_data = []
                x_hat_data.append([(0, 1), x_hat.clone().view(-1, 3072), y_hat.clone().view(-1)])  # 784 for mnist


            new_train_loader = self.train_loader.deepcopy()
            new_train_loader = new_train_loader[ind_task]
            print("before ", len(self.train_loader))
            new_train_loader.concatenate(data_loader.DataLoader(x_hat_data, self.args).increase_size(1))
            print('after ', len(new_train_loader))

            
            if self.args.method == 'Ractofit': #and ind_task > 1:
                print('<step 2> add randomly sampled data from previous GAN')
                self.model.eval()
                hard_samples = None
                z_rands = None
                for t in range(ind_task):
                    batch_sz = 5000 - self.args.num_z
                    lr, sz = 0.001, 1000
                    batches = math.ceil(batch_sz / sz)
                    for i in range(batches):
                        if self.args.dataset == 'mnist' or self.args.dataset == 'fashion':
                            Zs = Variable(torch.rand((sz, self.model.z_dim)).cuda(self.model.device), requires_grad=True)
                        else:
                            Zs = Variable(torch.randn((sz, self.model.z_dim)).cuda(self.model.device), requires_grad=True)
                        z_rand = copy.deepcopy(Zs)
                        if self.args.dataset == 'cifar100':
                            y = (torch.randperm(sz * 10) % ind_task * 10)[:sz]
                        elif self.args.dataset == 'imagenet50':
                            y = (torch.randperm(sz * 10) % ind_task * 50)[:sz]
                        else:
                            y = (torch.randperm(sz * 10) % ind_task)[:sz]
                        y_onehot = variable(self.model.get_one_hot(y), self.args.device)

                        optZ = optim.RMSprop([Zs], lr=lr)
                        for e in range(self.args.hardsample_epoch):
                            optZ.zero_grad()
                            if self.args.conditional == True:
                                current_score = self.model.D(self.model.G(Zs, y_onehot), y_onehot)
                            else:
                                current_score = self.model.D(self.model.G(Zs))
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
#                 save_images(hard_samples.data.numpy().transpose(0, 2, 3, 1)[:20 * 20, :, :, :], [20, 20],
#                             './forgettable_z/randomDGZ' + self.args.dataset + '_iter_' + str(self.args.hardsample_epoch) + '_task_' + str(
#                                 ind_task) + '.png')
#                 save_images(z_rands.data.numpy().transpose(0, 2, 3, 1)[:20 * 20, :, :, :], [20, 20],
#                             './forgettable_z/randomDGZ' + self.args.dataset + '_iter_' + str(self.args.hardsample_epoch) + '_task_' + str(
#                                 ind_task) + '_rand.png')
        
                # add hardsamples to train_loader
                print('before: ', len(new_train_loader[ind_task]))
                if self.args.conditional == False:
                    ys = torch.ones(hard_samples.shape[0], 1)
               
                hard_samples_loader = data_loader.DataLoader([[(0, 1), 
                       hard_samples.view(-1,self.input_size * self.image_size * self.image_size), ys.view(-1)]], self.args)
                new_train_loader.concatenate(hard_samples_loader)
                print('after: ', len(new_train_loader[ind_task]))


                
                

#             my_visualize_sample(new_train_loader, ind_task, self.args)
            new_train_loader.shuffle_task()

            if task_te_gen is not None:
                self.test_loader.concatenate(task_te_gen)
                test_loader = self.test_loader[ind_task]
                test_loader.shuffle_task()
            else:
                test_loader = None 
        else:
            train_loader = self.train_loader[ind_task]
            test_loader = self.test_loader[ind_task]

#             my_visualize_sample(train_loader, ind_task, self.args)
            train_loader.shuffle_task()

            return train_loader, test_loader

        return new_train_loader, test_loader

