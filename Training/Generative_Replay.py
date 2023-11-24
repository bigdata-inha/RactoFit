import os.path
import torch
from Training.Trainer import Trainer
import sys
sys.path.append(os.path.dirname(os.path.abspath('C:/Users/Jeonghyemin/PycharmProjects/Generative_Continual_Learning/Data/data_loader.py')))
from Data import data_loader
from utils import variable
import numpy as np
import math


class Generative_Replay(Trainer):
    def __init__(self, model, args):
        super(Generative_Replay, self).__init__(model, args)
                    
#     def create_next_data(self, ind_task):
#         ################## for fine-tune
#         return self.train_loader[ind_task], self.test_loader[ind_task]

#     def create_next_data(self, ind_task):
#         task_te_gen = None
#         if ind_task > 0:
#             self.train_loader[ind_task] # self.current_task를 ind_task로 설정.
#             train_loader = self.train_loader[ind_task]
#             batch_sz = 1000
#             batches = math.ceil(self.z_rehearsal.shape[0] / batch_sz)
#             print("z_rehearsal batches: ", batches)
#             for i in range(batches):
#                 start_idx = i * batch_sz
#                 end_idx = (i + 1) * batch_sz
#                 if i == batches - 1 and self.z_rehearsal.shape[0] % batch_sz != 0: end_idx = start_idx + self.z_rehearsal.shape[0] % batch_sz

#                 print("idx : ", start_idx, end_idx)
#                 with torch.no_grad():
#                     x_hat = self.model.G(variable(torch.Tensor(self.z_rehearsal[start_idx : end_idx])))
#                     self.model.expert.net.eval()
#                     y_hat = self.model.expert.labelize(x_hat, ind_task)
#                     x_hat = x_hat.cpu()
#                     y_hat = y_hat.cpu()
#                     x_hat_data = []
#                     x_hat_data.append([(0, 1), x_hat.clone().view(-1, 3072), y_hat.clone().view(-1)]) # 784 for mnist
#                     train_loader.concatenate(data_loader.DataLoader(x_hat_data, self.args))
#                     print("idx train_loader length", len(self.train_loader))
#             print('len(self.train_loader)', len(self.train_loader))
#             train_loader.shuffle_task()
    
    
#             if task_te_gen is not None:
#                 self.test_loader.concatenate(task_te_gen)
#                 test_loader = self.test_loader[ind_task]
#                 test_loader.shuffle_task()
#             else:
#                 test_loader = None #we don't use test loader for instance but we keep the code for later in case of
#         else:
#             train_loader = self.train_loader[ind_task]
#             test_loader = self.test_loader[ind_task]
#         return train_loader, test_loader

    def create_next_data(self, ind_task):
        task_te_gen = None
        if ind_task > 0:
            # 6742개의 기존 data
            self.train_loader[ind_task] # self.current_task를 ind_task로 설정.    we set the good index of dataset
            print("number of train sample per task is fixed as : " + str(self.sample_transfer))
            # 5000개 생성 : t-1시점까지의 data 생성.
            task_tr_gen = self.generate_dataset(ind_task, self.sample_transfer, classe2generate=ind_task, Train=True)

            print(len(self.train_loader))
            print(len(self.train_loader[ind_task]))
            self.train_loader.concatenate(task_tr_gen) # 6742개 + 5000개
            train_loader = self.train_loader[ind_task]
            print(len(self.train_loader))
            print(len(self.train_loader[ind_task]))

            train_loader.shuffle_task()

            if task_te_gen is not None:
                self.test_loader.concatenate(task_te_gen)
                test_loader = self.test_loader[ind_task]
                test_loader.shuffle_task()
            else:
                test_loader = None #we don't use test loader for instance but we keep the code for later in case of
        else:
            train_loader = self.train_loader[ind_task]
            test_loader = self.test_loader[ind_task]
        return train_loader, test_loader
