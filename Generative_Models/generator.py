import torch
import torch.nn as nn


def Generator(z_dim=62, dataset='mnist', conditional=False, model='VAE'):
    if dataset == 'mnist' or dataset == 'fashion':
        return MNIST_Generator(z_dim, dataset, conditional, model)
        # else:
        #    raise ValueError("This generator is not implemented")


class MNIST_Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim=62, dataset='mnist', conditional=False, model='VAE'):
        super(MNIST_Generator, self).__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.model = model
        self.conditional = conditional

        self.latent_dim = 1024

        self.input_height = 28
        self.input_width = 28
        self.input_dim = z_dim
        if self.conditional:
            self.input_dim += 10
        self.output_dim = 1

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.maxPool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.Sigmoid = nn.Sigmoid()
        #self.apply(self.weights_init)

    def forward(self, input, c=None):
        if c is not None:
            input = input.view(-1, self.input_dim - 10)
            input = torch.cat([input, c], 1)
        else:
            input = input.view(-1, self.input_dim)

        x = self.fc(input)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)
        return x

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class Generator_Cifar(nn.Module):
    def __init__(self, z_dim, conditional=False):
        super(Generator_Cifar, self).__init__()
        self.nc = 3
        self.conditional = conditional
        self.nz = z_dim
#         if self.conditional:
#             self.nz += 10
        self.ngf = 128 # generator dimension
        self.ndf = 128 # discriminator dimension
        self.ngpu = 1
        # if self.conditional: self.nz += 10 # option1
        if self.conditional : self.embedding = nn.Embedding(10, self.nz) # option2

        preprocess = nn.Sequential(
            nn.Linear(self.nz, 4 * 4 * 4 * self.ngf),
            nn.BatchNorm1d(4 * 4 * 4 * self.ngf),
            nn.ReLU(True)
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.ngf, 2 * self.ngf, 2, stride=2),
            nn.BatchNorm2d(2 * self.ngf),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.ngf, self.ngf, 2, stride=2),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(self.ngf, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input, c=None):
        if c is not None:
            # option1. embedding & multiple
            c_idx = torch.argmax(c, dim=1)
            c_embedding = self.embedding((c_idx))
#             print(input.shape)
            input = input.view(-1, self.nz)
            input = torch.mul(input, c_embedding)

            # # option2. concat
            # #print('G : before concat')
            # #print(input.shape)
            # input = input.view(-1, self.nz - 10)
            # #print(input.shape)
            # input = torch.cat([input, c], 1)
            # #print(input.shape)
            # input = input.view(-1, self.nz)
            # #print(input.shape)
            # # print('G input : ', input.shape)
            # # print('G input : ', input[:, self.nz-10:])
        else:
            input = input.view(-1, self.nz)
            #input = input.view(-1, self.nz, 1, 1)

        output = self.preprocess(input)
        output = output.view(-1, 4 * self.ngf, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)
