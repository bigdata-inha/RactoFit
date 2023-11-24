from utils import *
from log_utils import *
from tqdm import tqdm
from Data.data_loader import DataLoader
from Data.data_loader import *
import time
import numpy as np
from Evaluation.Reviewer import Reviewer
import gc
from Generative_Models.CWGAN_GP import CWGAN_GP
import torch.optim as optim
import math

class Trainer(object):
    def __init__(self, model, args, reviewer=None):
        self.args = args

        self.context = args.context

        if self.context == "Generation":
            self.reviewer = reviewer

        self.conditional = args.conditional
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.device = args.device
        self.verbose = args.verbose

        if self.dataset == "mnist" or self.dataset == "fashion":
            self.image_size = 28
            self.input_size = 1
        elif self.dataset == "cifar10" or self.dataset == "cifar100":
            self.image_size = 32
            self.input_size = 3

        self.model = model
        if self.args.gan_type == "GAN":
            self.previous_model = GAN(args)
            self.future_model = GAN(args)
        elif self.args.gan_type == "CGAN":
            self.previous_model = CGAN(args)
            self.future_model = CGAN(args)
        elif self.args.gan_type == "CWGAN_GP":
            self.previous_model = CWGAN_GP(args)
            self.future_model = CWGAN_GP(args)
        elif self.args.gan_type == "WGAN_GP":
            self.previous_model = WGAN_GP(args)
            self.future_model = WGAN_GP(args)
        else:
            print("GAN model is not exists")
        # self.model.load_G(0)
        # self.model.load_D(0)
        # self.previous_model.load_G(0)
        # self.previous_model.load_D(0)


        self.sample_dir = args.sample_dir
        self.sample_transfer = args.sample_transfer

        self.sample_num = 100
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.task_type = args.task_type
        self.method = args.method
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.epochs_gan = args.epoch_G
        self.task = None
        self.old_task = None
        self.num_task = args.num_task
        self.num_classes = args.num_classes
        self.ind_task = 0
        self.samples_per_task = args.samples_per_task

        train_loader, test_loader, n_inputs, n_outputs, n_tasks = load_datasets(args)
        self.train_loader = DataLoader(train_loader, args)
        self.test_loader = DataLoader(test_loader, args)

        self.z_rehearsal = np.array([])
        self.z_rehearsal_beforeDGZ = np.array([])
        self.train_iter = 0    

    def forward(self, x, ind_task):
        return self.model.net.forward(x)
    def additional_loss(self, model):
        return None
    def create_next_data(self, ind_task):
        return self.train_loader[ind_task], self.test_loader[ind_task]
    def preparation_4_task(self, model, ind_task):
        if ind_task > 0 and self.context == 'Generation':
            nb_sample_train = self.sample_transfer  # approximate size of one task
            nb_sample_test = int(nb_sample_train * 0.2)  # not used in fact

            # for FID evaluate
            self.model.generate_dataset(ind_task - 1, nb_sample_train, one_task=False, Train=True)
            self.model.generate_dataset(ind_task - 1, nb_sample_test, one_task=False, Train=False)

        train_loader, test_loader = self.create_next_data(ind_task)
        return train_loader, test_loader

    def update_z_rehearsal(self, ind_task, train_loader):   
        # step1. 
        print('before task zs, z_rehearsal.shape', self.z_rehearsal.shape)
        if self.z_rehearsal.size is not 0:
            print('<step1> update previous rehearsal')
            self.previous_model.G.eval()

            # label
            label = torch.tensor([i // self.args.num_z for i in range(ind_task*self.args.num_z)]) 
            if self.args.conditional:
                y_onehot = variable(self.model.get_one_hot(label), self.args.device)
                x_hat = self.previous_model.G(variable(torch.tensor(self.z_rehearsal), self.args.device), y_onehot)  
            else:
                x_hat = self.previous_model.G(variable(torch.tensor(self.z_rehearsal), self.args.device)) 
    
            n_iter = math.ceil(self.z_rehearsal.shape[0] / self.batch_size)
            for i in range (n_iter):
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                if i == n_iter - 1 and self.z_rehearsal.shape[0] % self.batch_size != 0: end_idx = start_idx + self.z_rehearsal.shape[0] % self.batch_size
                if self.args.conditional:
                    z_hat = self.model.conditional_find_z(x_hat[start_idx:end_idx], label[start_idx:end_idx], ind_task)
                else:
                    z_hat , _ = self.model.find_z(x_hat[start_idx:end_idx], ind_task)

                if i == 0: self.z_rehearsal = z_hat
                else: self.z_rehearsal = np.concatenate((self.z_rehearsal, z_hat))
            

    
    
        # step2.
        print('<step2> invert new task samples')
        print('train_loader len ', len(train_loader[ind_task]))
        for iter, (x_, t_) in enumerate(train_loader[ind_task]):
            print('iter', iter)
            if self.args.conditional:
                label = torch.tensor([ind_task] * self.batch_size) # new label
                if ind_task == 0:   new_z = self.model.conditional_find_z(x_, label, ind_task)
                else:   new_z = self.model.conditional_find_z(x_, label, ind_task) 
            else:
                if ind_task == 0:   new_z, _ = self.model.find_z(x_, ind_task)
                else:   new_z, _ = self.model.find_z(x_, ind_task) 
            
            n_iter = int(self.args.num_z / self.batch_size)
            remain = self.args.num_z % self.batch_size
            print('n_iter ', n_iter, ' remain ', remain)

            if iter == n_iter: 
                if self.z_rehearsal.shape[0] == 0: self.z_rehearsal = new_z[:remain]
                else: self.z_rehearsal = np.concatenate((self.z_rehearsal, new_z[:remain]))
                break
            elif self.z_rehearsal.shape[0] == 0: self.z_rehearsal = new_z
            else: self.z_rehearsal = np.concatenate((self.z_rehearsal, new_z))
        print('final z_rehearsal.shape', self.z_rehearsal.shape)

        
      
        
    def run_generation_tasks(self):
        # run each task for a model
        # self.model.G.apply(self.model.G.weights_init)
        for ind_task in range(0, self.args.num_task):
            print("############################ Task : " + str(ind_task))

            # prepare training data
            if 'Ewc' in self.method or 'D_Ewc' in self.method:
                train_loader, test_loader = self.preparation_4_task(self.model, ind_task)
            else:
                train_loader, test_loader = self.preparation_4_task(self.model.G, ind_task)
            self.ind_task = ind_task


            # train
            print('\n')
            print('<<start training>>')
                
                
            self.train_iter = 0    
            for epoch in range(self.args.epochs): # train
                print("Epoch : " + str(epoch))
                loss_epoch, now_train_iter = self.model.train_on_task(train_loader, ind_task, epoch, self.additional_loss, self.train_iter)
                self.train_iter += now_train_iter
                self.model.visualize_results((epoch + 1), ind_task) 

            if self.args.method == 'Ractofit':
                if ind_task < self.args.num_task-1:
                    print('\n')
                    print('<<prepare memory>>')
                    self.update_z_rehearsal(ind_task, self.train_loader)
                    self.model.save_G(self.ind_task)
                    self.previous_model.load_G(self.ind_task)


            self.model.save_G(self.ind_task)
            self.model.save_D(self.ind_task)



        nb_sample_train = self.sample_transfer 
        self.model.generate_dataset(self.num_task - 1, nb_sample_train, one_task=False, Train=True)

    def run_classification_tasks(self):
        accuracy_test = 0
        loss, acc, acc_all_tasks = {}, {}, {}
        for ind_task in range(self.num_task):
            accuracy_task = 0
            train_loader, test_loader = self.preparation_4_task(self.model.net, ind_task)

            self.ind_task = ind_task

            if not self.args.task_type == "CUB200":
                path = os.path.join(self.sample_dir, 'sample_' + str(ind_task) + '.png')

                if self.verbose:
                    print("some sample from the train_loader")
                self.train_loader.visualize_sample(path, self.sample_num,
                                                   [self.image_size, self.image_size, self.input_size])
            else:
                print("visualisation of CUB200 not implemented")
            loss[ind_task] = []
            acc[ind_task] = []
            acc_all_tasks[ind_task] = []
            for epoch in tqdm(range(self.args.epochs)):
                loss_epoch, accuracy_epoch = self.model.train_on_task(train_loader, ind_task, epoch,
                                                                      self.additional_loss)
                loss[ind_task].append(loss_epoch)

                if accuracy_epoch > accuracy_task:
                    self.model.save(ind_task)
                    accuracy_task = accuracy_epoch

                for previous_task in range(ind_task + 1):
                    loss_test, test_acc, classe_prediction, classe_total, classe_wrong = self.model.eval_on_task(
                        self.test_loader[previous_task], 0)

                    # acc[previous_task].append(self.test(previous_task))
                    acc[previous_task].append(test_acc)

                accuracy_test_epoch = self.test_all_tasks()
                acc_all_tasks[ind_task].append(accuracy_test_epoch)

                if accuracy_test_epoch > accuracy_test:
                    # if True:
                    self.model.save(ind_task, Best=True)
                    accuracy_test = accuracy_test_epoch
        loss_plot(loss, self.args)
        accuracy_plot(acc, self.args)
        accuracy_all_plot(acc_all_tasks, self.args)
    def test_all_tasks(self):
        self.model.net.eval()

        mean_task = 0
        if self.task_type == 'upperbound':
            loss, mean_task, classe_prediction, classe_total, classe_wrong = self.model.eval_on_task(
                self.test_loader[self.num_task - 1], 0)
        else:
            for ind_task in range(self.num_task):
                loss, acc_task, classe_prediction, classe_total, classe_wrong = self.model.eval_on_task(
                    self.test_loader[ind_task], 0)

                mean_task += acc_task
            mean_task = mean_task / self.num_task
        print("Mean overall performance : " + str(mean_task.item()))
        return mean_task
    def regenerate_datasets_for_eval(self):
        nb_sample_train = self.sample_transfer  # len(self.train_loader[0])
        # nb_sample_test = int(nb_sample_train * 0.2)

        for i in range(self.args.num_task):
            self.model.load_G(ind_task=i)
            self.generate_dataset(i, nb_sample_train, classe2generate=i + 1, Train=True)
        return
    def generate_dataset(self, ind_task, sample_per_classes, classe2generate, Train=True):
        return self.model.generate_dataset(ind_task, sample_per_classes, one_task=False, Train=Train,
                                           classe2generate=classe2generate)
