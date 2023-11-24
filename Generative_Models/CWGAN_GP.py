import utils, torch, time
import numpy as np
from torch.autograd import Variable
from torch.autograd import grad
from Generative_Models.Conditional_Model import ConditionalModel
from Data.load_dataset import get_iter_dataset
from torch.utils.data import DataLoader

from utils import variable
import math


class CWGAN_GP(ConditionalModel):
    def __init__(self, args):
        super(CWGAN_GP, self).__init__(args)
        self.model_name = 'CWGAN_GP'
        self.lambda_gp = 10 
        self.cuda = True
        self.n_critic = 5
        self.Tensor = torch.cuda.FloatTensor if True else torch.FloatTensor

    def dis_hinge(self, dis_fake, dis_real):
        loss = torch.mean(torch.relu(1. - dis_real)) + \
               torch.mean(torch.relu(1. + dis_fake))
        return loss

    def gen_hinge(self, dis_fake, dis_real=None):
        return -torch.mean(dis_fake)
    
    def decay_lr(self, opt, max_iter, start_iter, initial_lr):
        """Decay learning rate linearly till 0."""
        coeff = -initial_lr / (max_iter - start_iter)
        for pg in opt.param_groups:
            pg['lr'] += coeff
    def run_batch(self, x_, t_, iter, train_iter, additional_loss=None):

        wgangp = True # False: hinge loss
        
    
            
        for p in self.D.parameters(): p.requires_grad = True
        self.G.train()
        self.D.train()

        x_ = variable(x_, self.args.device)
        x_ = x_.view(-1, self.input_size, self.size, self.size)  # cifar10
        y_onehot = variable(self.get_one_hot(t_), self.args.device)
        z_ = variable(self.random_tensor(x_.size(0), self.z_dim), self.args.device)  # uniform dist

        # update D network
        self.D_optimizer.zero_grad()
        D_real = self.D(x_, y_onehot)
        if wgangp:
            D_real_loss = D_real.mean().view(-1)
        fake = self.G(z_, y_onehot).detach() 
        D_fake = self.D(fake, y_onehot)
        if wgangp:
            D_fake_loss = D_fake.mean().view(-1)

            # gradient penalty 
            if self.gpu_mode:
                alpha = torch.rand((x_.size(0), 1)).cuda(self.device)
            else:
                alpha = torch.rand((x_.size(0), 1, 1, 1))
            alpha = alpha.expand(x_.size(0), self.input_size * self.size * self.size).contiguous().view(x_.size(0),
                                                                    self.input_size, self.size, self.size)
            x_hat = Variable(alpha * x_.data + (1 - alpha) * fake.data, requires_grad=True)
            if self.gpu_mode: x_hat = x_hat.cuda(self.device)
            pred_hat = self.D(x_hat.view(-1, self.input_size, self.size, self.size), y_onehot)
            gradients = grad(outputs=pred_hat, inputs=x_hat,
                             grad_outputs=torch.ones(pred_hat.size()).cuda(self.device) if self.gpu_mode else torch.ones(
                                 pred_hat.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
            D_loss = D_fake_loss - D_real_loss + gradient_penalty
            # Wasserstein_D = D_real_loss - D_fake_loss
        else: # hinge loss
            D_loss = self.dis_hinge(D_fake, D_real)
        
        D_loss.backward()
        self.D_optimizer.step()




        # update G network
        if ((iter + 1) % self.n_critic) == 0:
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation
            self.G_optimizer.zero_grad()

            z_ = variable(self.random_tensor(x_.size(0), self.z_dim), self.args.device)
            fake = self.G(z_, y_onehot)
            D_fake = self.D(fake, y_onehot)
            if wgangp:
                G_loss = -torch.mean(D_fake)
            else: # hinge loss
                G_loss = self.gen_hinge(D_fake)
            G_loss.backward()
            self.G_optimizer.step()
            # print('loss ', G_loss.item(), D_loss.item(), Wasserstein_D.item())

    def train_on_task(self, train_loader, ind_task, epoch, additional_loss, train_iter):
        self.G.train()
        self.D.train()
        epoch_start_time = time.time()
        sum_loss_train = 0.
        Wasserstein_costs = []
        
        now_train_iter = 0
        
        for iter, (x_, t_) in enumerate(train_loader):
            self.run_batch(x_, t_, iter, train_iter, additional_loss=None)
            now_train_iter += 1

        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        # self.save()

        sum_loss_train = sum_loss_train / float(len(train_loader))
        return sum_loss_train, now_train_iter