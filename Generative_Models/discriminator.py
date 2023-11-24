import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, dataset='mnist', conditional=False, model='VAE'):
        super(Discriminator, self).__init__()
        self.dataset = dataset
        self.model = model
        self.conditional = conditional
        if dataset == 'mnist' or dataset == 'fashion':
            self.input_height = 28
            self.input_width = 28
            self.input_dim = 1
            self.output_dim = 1

        self.latent_dim = 1024

        shape = 128 * (self.input_height // 4) * (self.input_width // 4)

        self.fc1_1 = nn.Linear(784, self.latent_dim)
        self.fc1_2 = nn.Linear(10, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim * 2, self.latent_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(self.latent_dim // 2)
        self.fc3 = nn.Linear(self.latent_dim // 2, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(shape, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.latent_dim, self.output_dim),
            nn.Sigmoid(),
        )
        self.aux_linear = nn.Linear(shape, 10)
        self.softmax = nn.Softmax()
        #self.apply(self.weights_init)

        if self.model == 'BEGAN':
            self.be_conv = nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1),
                nn.ReLU(),
            )
            self.be_fc = nn.Sequential(
                nn.Linear(64 * (self.input_height // 2) * (self.input_width // 2), 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 64 * (self.input_height // 2) * (self.input_width // 2)),
                nn.BatchNorm1d(64 * (self.input_height // 2) * (self.input_width // 2)),
                nn.ReLU(),
            )
            self.be_deconv = nn.Sequential(
                nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
                # nn.Sigmoid(),
            )
            utils.initialize_weights(self)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def disc_cgan(self, input, label):
        input = input.view(-1, self.input_height * self.input_width * self.input_dim)
        x = F.leaky_relu(self.fc1_1(input), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))
        return x, label

    def disc_began(self, input):
        x = self.be_conv(input)
        x = x.view(x.size()[0], -1)
        x = self.be_fc(x)
        x = x.view(-1, 64, (self.input_height // 2), (self.input_width // 2))
        x = self.be_deconv(x)
        return x

    def forward(self, input, c=None):
        # print(input.data.shape)
        if self.model == 'BEGAN':
            return self.disc_began(input)

        if self.model == "CGAN" or (self.model == 'GAN' and self.conditional):  # CGAN
            return self.disc_cgan(input, c)

        x = self.conv(input)
        x = x.view(x.data.shape[0], 128 * (self.input_height // 4) * (self.input_width // 4))

        final = self.fc(x)
        if c is not None:
            c = self.aux_linear(x)
            c = self.softmax(c)
            return final, c
        else:
            return final

class Discriminator_Cifar(nn.Module):
    def __init__(self, conditional):
        super(Discriminator_Cifar, self).__init__()
        self.nc = 3
        self.ndf = 128
        self.ngpu = 1
        self.conditional = conditional
        if self.conditional : self.embedding = nn.Embedding(10, 4 * 4 * 4 * self.ndf)


        main = nn.Sequential(
            nn.Conv2d(3, self.ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.ndf, 2 * self.ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.ndf, 4 * self.ndf, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * self.ndf, 1)

    def forward(self, input, c=None):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 4 * self.ndf)

        if c is not None:
            # option1. embedding & multiple
            c_idx = torch.argmax(c, dim=1)
            output = torch.mul(output, self.embedding(c_idx))
            output = self.linear(output)
            # # option2. concat
            # #print('D : before concat')
            # #print(output.shape)
            # output = torch.cat([output, c], 1)
            # # print('D output : ', output.shape)
            # # print('D output : ', output[:, -10:])
            # #print(output.shape)
            # output = self.linearC(output)
            # #output = F.sigmoid(output)
        else:
            output = self.linear(output)
        return output  