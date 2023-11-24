import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

import utils
from Classifiers.Cifar_Classifier import Cifar_Classifier
from Data.load_dataset import load_dataset_full, load_dataset_test, get_iter_dataset
from log_utils import *
from Data.data_loader import *
from Evaluation.tools import calculate_frechet_distance
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):

        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


class Reviewer(object):
    def __init__(self, args):
        # parameters
        self.args = args
        self.epoch_Review = args.epoch_Review
        self.sample_num = 64
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.sample_dir = args.sample_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.data_dir = args.data_dir
        self.gen_dir = args.gen_dir
        self.verbose = args.verbose

        self.lr = args.lrC
        self.momentum = args.momentum
        self.log_interval = 100
        self.sample_num = 100
        self.size_epoch = args.size_epoch
        self.gan_type = args.gan_type
        self.conditional = args.conditional
        self.device = args.device
        self.trainEval = args.trainEval
        self.num_task = args.num_task
        self.task_type = args.task_type
        self.context = args.context

        self.seed = args.seed

        if self.conditional:
            self.model_name = 'C' + self.model_name

        # Load the generator parameters

        # The reviewer evaluate generate dataset (loader train) on true data (loader test)
        # not sur yet if valid should be real or not (it was before)
        dataset_train, dataset_valid, list_class_train, list_class_valid = load_dataset_full(self.data_dir,
                                                                                             args.dataset)
        dataset_test, list_class_test = load_dataset_test(self.data_dir, args.dataset, args.batch_size)

        # create data loader for validation and testing
        self.valid_loader = get_iter_dataset(dataset_valid)
        self.test_loader = get_iter_dataset(dataset_test)

        if self.dataset == 'mnist':
            self.input_size = 1
            self.size = 28
        elif self.dataset == 'fashion':
            self.input_size = 1
            self.size = 28
        elif self.dataset == 'cifar10' or self.dataset == 'mydata':
            self.input_size = 3
            self.size = 32

        if self.dataset == 'mnist':
            self.Classifier = Mnist_Classifier(args)
        elif self.dataset == 'fashion':
            self.Classifier = Fashion_Classifier(args)
        elif self.dataset == 'cifar10' or self.dataset == 'mydata':
            self.Classifier = Cifar_Classifier(args)
        elif self.dataset == 'cifar100':
            self.Classifier = Cifar_Classifier(args, num_classes=100)
        else:
            print('Not implemented')
    def save(self, best=False):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if best:
            torch.save(self.Classifier.net.state_dict(),
                       os.path.join(self.save_dir, self.model_name + '_Classifier_Best.pkl'))
        else:
            torch.save(self.Classifier.net.state_dict(),
                       os.path.join(self.save_dir, self.model_name + '_Classifier.pkl'))
    def load(self, reference=False):
        if reference:
            save_dir = os.path.join(self.save_dir, "..", "..", "..", "Classifier", 'seed_' + str(self.seed))
            self.Classifier.net.load_state_dict(torch.load(os.path.join(save_dir, 'Classifier_Classifier_Best.pkl')))
        else:
            self.Classifier.net.load_state_dict(
                torch.load(os.path.join(self.save_dir,
                                        self.model_name + '_Classifier_Best.pkl')))
    def load_best_baseline(self):
        best_seed = utils.get_best_baseline(self.log_dir, self.dataset)

        save_dir = os.path.join(self.save_dir, "..", "..", "..", "Classifier", 'seed_' + str(best_seed))
        self.Classifier.net.load_state_dict(torch.load(os.path.join(save_dir, 'Classifier_Classifier_Best.pkl')))
    def train_classifier(self, epoch, data_loader_train, ind_task):  # this should be train on task
        self.Classifier.net.train()
        train_loss_classif, train_accuracy = self.Classifier.train_on_task(data_loader_train, ind_task=ind_task,
                                                                           epoch=epoch, additional_loss=None)
        val_loss_classif, valid_accuracy, classe_prediction, classe_total, classe_wrong = self.Classifier.eval_on_task(
            self.valid_loader, self.verbose)

        if self.verbose:
            print(
                'Epoch: {} Train set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n Valid set: Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(
                    epoch, train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy))
        return train_loss_classif, train_accuracy, val_loss_classif, valid_accuracy, (
                    100. * classe_prediction) / classe_total
    def visualize_dataloader(self, dataloader):
        for i, (data, target) in enumerate(dataloader):
            print(i, data.shape)
            x = data.view((-1, 3, 32, 32))
            image_frame_dim = int(np.floor(np.sqrt(x.size(0))))
            x = x.cpu().data.numpy().transpose(0, 2, 3, 1)
            save_images(x[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        './visualize_test/' + 'iter' + str(i) + '.png')

    def compute_all_tasks_FID(self, args, Best=False):
        if Best: id = "Best_"
        else: id = ''
        list_FID = []
        list_generatedNum = []

        fid = self.compute_FID(args, 9, Best)
        print(fid)
      
    def compute_FID(self, args, ind_task, Best=False):
        if Best: id = "Best_"
        else: id = ''
        if 'upperbound' in self.task_type: test_file = self.task_type + '_' + str(self.num_task) + '_train.pt' # '_test.pt'
        else: test_file = 'upperbound_' + self.task_type + '_' + str(self.num_task) + '_train.pt' # '_test.pt'

            
        print(os.path.join(self.data_dir, 'Tasks', self.dataset, test_file))
        true_DataLoader = DataLoader(torch.load(os.path.join(self.data_dir, 'Tasks', self.dataset, test_file)), args)[9]
        gen_DataLoader = DataLoader(torch.load(os.path.join(self.gen_dir, id + 'train_Task_' + str(9) + '.pt')), args)
        # self.visualize_dataloader(gen_DataLoader)
        # self.visualize_dataloader(true_DataLoader)
        fid = self.Frechet_Inception_Distance(gen_DataLoader, true_DataLoader, ind_task)
        

#         true_DataLoader = DataLoader(torch.load(os.path.join(self.data_dir, 'Tasks', self.dataset, test_file)), args)[9]
#         gen_DataLoader = DataLoader(torch.load(os.path.join(self.gen_dir, id + 'train_Task_' + str(9) + '.pt')), args)
#         fid_incep = self.Inceptionv3_Frechet_Inception_Distance(gen_DataLoader, true_DataLoader, ind_task)
#         print(fid_incep)
        return fid



    def Inceptionv3_Frechet_Inception_Distance(self, Gen_DataLoader, True_DataLoader, ind_task):
        if ind_task == 0: eval_size = 77
        else: eval_size = 77 * (ind_task)   # 45000(44352)개
        eval_size = 194

        if self.dataset == "mnist":
            latent_size = 320
        elif self.dataset == "fashion":
            latent_size = 320
        elif self.dataset == "cifar10":
            latent_size = 2048 


        # inceptionv3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx])
        model = model.cuda(self.device)
        model.eval()


        real_output_table = np.empty((eval_size * self.batch_size, latent_size), dtype='float')
        gen_output_table = np.empty((eval_size * self.batch_size, latent_size), dtype='float')  # 3200(50*64) * 2048
        True_DataLoader.shuffle_task()
        for i, (data, target) in enumerate(True_DataLoader):
            if i >= eval_size or i >= (int(len(True_DataLoader) / self.batch_size) - 1): break
            if self.dataset == 'mnist': x_ = data.view((-1, 28, 28))
            else: x_ = data.view((-1, 3, 32, 32))
            if self.gpu_mode:
                data, target = x_.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            # print(batch.shape)  # 64*32*32
            activation = np.squeeze(model(batch)[0].cpu().data.numpy())
            # print(activation.shape) # 64*2048
            real_output_table[i * self.batch_size:(i + 1) * self.batch_size, :] = activation


        Gen_DataLoader.shuffle_task()
        for i, (data, target) in enumerate(Gen_DataLoader):
            if i >= eval_size or i >= (int(len(Gen_DataLoader) / self.batch_size) - 1): break
            if self.dataset == 'mnist': x_ = data.view((-1, 28, 28))
            else:  x_ = data.view((-1, 3, 32, 32))
            if self.gpu_mode:
                data, target = x_.cuda(self.device), target.cuda(self.device)
            batch = Variable(data)
            activation = np.squeeze(model(batch)[0].cpu().data.numpy())
            gen_output_table[i * self.batch_size:(i + 1) * self.batch_size, :] = activation

        # compute mu_real and sigma_real
        mu_real = real_output_table.mean(0)
        cov_real = np.cov(real_output_table.transpose())
        # print(mu_real.shape)
        # print(cov_real.shape)
        assert mu_real.shape[0] == latent_size
        assert cov_real.shape[0] == cov_real.shape[1] == latent_size
        mu_gen = gen_output_table.mean(0)
        cov_gen = np.cov(gen_output_table.transpose())
        assert mu_gen.shape[0] == latent_size
        assert cov_gen.shape[0] == cov_gen.shape[1] == latent_size

        Frechet_Inception_Distance = calculate_frechet_distance(mu_real, cov_real, mu_gen, cov_gen)
        print(Frechet_Inception_Distance)
        return Frechet_Inception_Distance
    def Frechet_Inception_Distance(self, Gen_DataLoader, True_DataLoader, ind_task):
        if ind_task == 0: eval_size = 77
        else: eval_size = 77 * (ind_task) 
        eval_size = 194
        if self.dataset == "mnist":
            latent_size = 320
        elif self.dataset == "fashion":
            latent_size = 320
        elif self.dataset == "cifar10":
            latent_size = 512 #2432
            
        # with pre-trained ResNet18
        self.Classifier = resnet18(pretrained=True).cuda(self.device)
        self.Classifier.eval()
        True_DataLoader.shuffle_task()
        for i, (data, target) in enumerate(True_DataLoader):
            if i >= eval_size or i >= (int(len(True_DataLoader) / self.batch_size) - 1):  # throw away the last batch
                break
            if self.gpu_mode: data, target = data.cuda(self.device), target.cuda(self.device)
            data = data.view(-1, 3, 32, 32)
            batch = Variable(data)
            label = Variable(target.squeeze())
            activation = self.Classifier(batch) # batchsz * 512
            
            if i == 0: real_output_table = activation.data
            else: real_output_table = torch.cat([real_output_table, activation.data])

        Gen_DataLoader.shuffle_task()
        for i, (data, target) in enumerate(Gen_DataLoader):
            if i >= eval_size or i >= (
                        int(len(Gen_DataLoader) / self.batch_size) - 1):  # (we throw away the last batch)
                break
            # 2. use the reference classifier to compute the output vector
            if self.gpu_mode: data, target = data.cuda(self.device), target.cuda(self.device)
            data = data.view(-1, 3, 32, 32)
            batch = Variable(data)
            data, target = data.cuda(self.device), target.cuda(self.device)
            label = Variable(target.squeeze())
            activation = self.Classifier(batch)
            if i == 0: gen_output_table = activation.data
            else: gen_output_table = torch.cat([gen_output_table, activation.data])
#         print('task', ind_task, ': real', real_output_table.shape, ', gen', gen_output_table.shape)

        # compute mu_real and sigma_real
        mu_real = real_output_table.cpu().numpy().mean(0)
        cov_real = np.cov(real_output_table.cpu().numpy().transpose())
        assert mu_real.shape[0] == latent_size
        assert cov_real.shape[0] == cov_real.shape[1] == latent_size
        mu_gen = gen_output_table.cpu().numpy().mean(0)
        cov_gen = np.cov(gen_output_table.cpu().numpy().transpose())
        assert mu_gen.shape[0] == latent_size
        assert cov_gen.shape[0] == cov_gen.shape[1] == latent_size

        Frechet_Inception_Distance = calculate_frechet_distance(mu_real, cov_real, mu_gen, cov_gen)
        return Frechet_Inception_Distance


    def eval_on_train(self, data_loader_train, task):
        if self.dataset == 'mnist':
            self.Classifier = Mnist_Classifier(self.args)
        elif self.dataset == 'fashion':
            self.Classifier = Fashion_Classifier(self.args)
        elif self.dataset == 'cifar10' or self.dataset == 'mydata':
            self.Classifier = Cifar_Classifier(self.args)
        else:
            print('Not implemented')
        self.Classifier.load_expert()
        self.Classifier.net.eval()
        print("trainEval Task : " + str(task))

        loss, train_acc, classe_prediction, classe_total, classe_wrong = self.Classifier.eval_on_task(data_loader_train,
                                                                                                      self.verbose)
        train_acc_classes = 100. * classe_prediction / classe_total

        if self.verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy : ({:.2f}%)'.format(
                loss, train_acc))
            for i in range(10):
                print('Classe {} Accuracy: {}/{} ({:.3f}%, Wrong : {})'.format(
                    i, classe_prediction[i], classe_total[i],
                    100. * classe_prediction[i] / classe_total[i], classe_wrong[i]))
            print('\n')

        return train_acc, train_acc_classes
    def eval_balanced_on_train(self, data_loader_train):
        cpt_classes = np.zeros(10)
        for i, (data, target) in enumerate(data_loader_train):
            for i in range(target.shape[0]):
                cpt_classes[target[i]] += 1
        print(cpt_classes.astype(int))
        return cpt_classes.astype(int)
    def review_all_tasks(self, args, Best=False):
        # before launching the programme we check that all files are here to not lose time
        
        per_task = False
        if per_task == True: start_idx = 0
        else: start_idx = 9
        
        per_task_list = []
        for i in range(start_idx, self.num_task):  ##################
            if Best: path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else: path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')
            assert os.path.isfile(path)

        for i in range(start_idx, self.num_task):  ##################
            if Best:  path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else: path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')
            print(self.gen_dir, 'train_Task_' + str(i) + '.pt')
            data_loader_train = DataLoader(torch.load(path), args)
            acc = self.review(data_loader_train, i, per_task, Best)
            per_task_list.append(acc)
        print(per_task_list)
    def review(self, data_loader_train, task, per_task, Best=False):
        if Best: id = "Best_"
        else: id = ''
        if self.dataset == 'mnist':
            self.Classifier = Mnist_Classifier(self.args)
        elif self.dataset == 'fashion':
            self.Classifier = Fashion_Classifier(self.args)
        elif self.dataset == 'cifar10' or self.dataset == 'mydata':
            self.Classifier = Cifar_Classifier(self.args)
        else: print('Not implemented')

        best_accuracy = -1
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        valid_acc = []
        valid_acc_classes = []

#         self.visualize_dataloader(data_loader_train)
#         self.visualize_dataloader(self.test_loader)
        print("Task : " + str(task))
        early_stop = 0.
        # Training classifier
        for epoch in range(self.epoch_Review):
            tr_loss, tr_acc, v_loss, v_acc, v_acc_classes = self.train_classifier(epoch, data_loader_train, task)
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            # Save best model
            if v_acc > best_accuracy:
                if self.verbose:
                    print("New Best Classifier")
                    print(v_acc)
                best_accuracy = v_acc
                self.save(best=True)
                early_stop = 0.
            if early_stop == 60: break
            else: early_stop += 1
            valid_acc.append(np.array(v_acc))
            valid_acc_classes.append(np.array(v_acc_classes))
            print('train_acc', tr_acc)
            print('valid_acc', v_acc)

        self.load()
        loss, test_acc, classe_prediction, classe_total, classe_wrong = self.Classifier.eval_on_task(self.test_loader, self.verbose)
        test_acc_classes = 100. * classe_prediction / classe_total
        list = []
        list.append(test_acc)
        
            
        
        if per_task == True:
            final_acc = 0
            for i in range(task+1):
                print(i, classe_prediction[i]/10)
                final_acc += classe_prediction[i]/10
            print ('per_task' + str(task) + ' acc: ' + str(final_acc / (task+1)))
            return final_acc / (task+1)
        else:
            print('test_acc', test_acc)
            print('\nTest set: Average loss: {:.4f}, Accuracy : ({:.2f}%)'.format(loss, test_acc))
            for i in range(10):
                print('Classe {} Accuracy: {}/{} ({:.3f}%, Wrong : {})'.format(
                    i, classe_prediction[i], classe_total[i], 100. * classe_prediction[i] / classe_total[i], classe_wrong[i]))
            for i in range(10):
                print(classe_prediction[i] / 10)
                list.append(classe_prediction[i] / 10)
            

        np.savetxt('./Eval/' + str(self.args.method) + '_hardepoch' + str(
            self.args.hardsample_epoch) + '_seed' + str(self.args.seed) + '_Fitting_capacity.txt', np.transpose(list))

    def review_all_trainEval(self, args, Best=False):
        if Best:
            id = "Best_"
        else:
            id = ''
        list_trainEval = []
        list_trainEval_classes = []
        list_balance_classes = []

        # before launching the programme we check that all files are here to not lose time
        for i in range(self.num_task):
            if Best:  # Best can be use only for Baseline
                path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else:
                path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')
                print(path)
            assert os.path.isfile(path)

        for i in range(self.num_task):
            if Best:  # Best can be use only for Baseline
                path = os.path.join(self.gen_dir, 'Best_train_Task_' + str(i) + '.pt')
            else:
                path = os.path.join(self.gen_dir, 'train_Task_' + str(i) + '.pt')

            data_loader_train = DataLoader(torch.load(path), args)

            if self.conditional or Best:
                train_acc, train_acc_classes = self.eval_on_train(data_loader_train, self.verbose)
                list_trainEval.append(train_acc)
                list_trainEval_classes.append(train_acc_classes)
            else:
                classe_balance = self.eval_balanced_on_train(data_loader_train)
                list_balance_classes.append(classe_balance)

        if self.conditional or Best:
            assert len(list_trainEval) == self.num_task

            list_trainEval = np.array(list_trainEval)
            list_trainEval_classes = np.array(list_trainEval)

            np.savetxt(os.path.join(self.log_dir, id + 'TrainEval_All_Tasks.txt'), list_trainEval)
            np.savetxt(os.path.join(self.log_dir, id + 'TrainEval_classes_All_Tasks.txt'), list_trainEval_classes)
        else:
            assert len(list_balance_classes) == self.num_task
            np.savetxt(os.path.join(self.log_dir, id + 'Balance_classes_All_Tasks.txt'), list_balance_classes)
    # save a classifier or the best classifier