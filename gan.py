import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class VideoDCD(nn.Module):
    '''
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size, time_steps, x_h=64, x_w=64, filter_size=128, j=16, nchannel=1, bn=False):
        super(VideoDCD, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.filter_size = filter_size
        self.nchannel = nchannel
        self.ks = 6
        # j is the dimension of h and M
        self.j = j
        self.bn = bn
        self.x_height = x_h
        self.x_width = x_w

        h_in = 8
        s = 2
        p = self.compute_padding(h_in, s, self.ks)

        conv_layers = [nn.Conv2d(self.nchannel, self.filter_size, kernel_size=[self.ks, self.ks],
                                 stride=[s, s], padding=[p, p])]
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size))
        conv_layers.append(nn.LeakyReLU())
        conv_layers.append(nn.Conv2d(self.filter_size, self.filter_size*2, kernel_size=[self.ks, self.ks],
                                     stride=[s, s], padding=[p, p]))
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size * 2))
        conv_layers.append(nn.LeakyReLU())
        conv_layers.append(nn.Conv2d(self.filter_size*2, self.filter_size*4, kernel_size=[self.ks, self.ks],
                                     stride=[s, s], padding=[p, p]))
        if self.bn:
            conv_layers.append(nn.BatchNorm2d(self.filter_size*4))
        conv_layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*conv_layers)

        self.lstm1 = nn.LSTM(self.filter_size * 4 * 8 * 8, self.filter_size*4, batch_first=True)
        self.lstmbn = nn.BatchNorm1d(self.filter_size*4)
        self.lstm2 = nn.LSTM(self.filter_size*4, self.j, batch_first=True)
        # self.sig = nn.Sigmoid()

    # padding computation when h_in = 2h_out
    def compute_padding(self, h_in, s, k_size):
        return max((h_in * (s - 2) - s + k_size) // 2, 0)

    def forward(self, inputs):
        x = inputs.reshape(self.batch_size * self.time_steps, self.nchannel, self.x_height, self.x_width)
        x = self.conv_net(x)
        x = x.reshape(self.batch_size, self.time_steps, -1)
        # first output dimension is the sequence of h_t.
        # second output is h_T and c_T(last cell state at t=T).
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        # x = self.sig(x)
        return x


class VideoDCG(nn.Module):
    '''
    Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the model has shape [batch_size, x_height, x_weight*time_step, channel]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, batch_size=8, time_steps=32, x_h=64, x_w=64, filter_size=32, state_size=32, nchannel=1, z_dim=25,
                 y_dim=20, bn=False, output_act='sigmoid'):
        super(VideoDCG, self).__init__()

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.filter_size = filter_size
        self.state_size = state_size
        self.nchannel = nchannel
        self.n_noise_t = z_dim
        self.n_noise_y = y_dim
        self.x_height = x_h
        self.x_width = x_w
        self.bn = bn
        self.output_activation = output_act

        self.lstm1 = nn.LSTM(self.n_noise_t + self.n_noise_y, self.state_size, batch_first=True)
        self.lstmbn1 = nn.BatchNorm1d(self.state_size)
        self.lstm2 = nn.LSTM(self.state_size, self.state_size*2, batch_first=True)
        self.lstmbn2 = nn.BatchNorm1d(self.state_size*2)

        dense_layers = [nn.Linear(self.state_size*2, 8*8*self.filter_size*4)]
        if self.bn:
            dense_layers.append(nn.BatchNorm1d(8*8*self.filter_size*4))
        dense_layers.append(nn.LeakyReLU())

        self.dense_net = nn.Sequential(*dense_layers)

        h_in = 8
        s = 2
        k_size = 6
        p = self.compute_padding(h_in, s, k_size)

        deconv_layers = [nn.ConvTranspose2d(self.filter_size*4, self.filter_size*4, kernel_size=[k_size, k_size],
                                            stride=[s, s], padding=[p, p])]
        
        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size*4))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(nn.ConvTranspose2d(self.filter_size * 4, self.filter_size * 2,
                                                kernel_size=[k_size, k_size], stride=[s, s], padding=[p, p]))

        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size * 2))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(nn.ConvTranspose2d(self.filter_size * 2, self.filter_size,
                                                kernel_size=[6, 6], stride=[2, 2], padding=[p, p]))
        if self.bn:
            deconv_layers.append(nn.BatchNorm2d(self.filter_size))
        deconv_layers.append(nn.LeakyReLU())

        deconv_layers.append(nn.ConvTranspose2d(self.filter_size, self.nchannel, kernel_size=[5, 5],
                                                stride=[1, 1], padding=[2, 2]))

        if self.output_activation == 'sigmoid':
            deconv_layers.append(nn.Sigmoid())
        elif self.output_activation == 'tanh':
            deconv_layers.append(nn.Tanh())
        else:
            deconv_layers = deconv_layers

        self.deconv_net = nn.Sequential(*deconv_layers)

    # padding computation when 2h_in = h_out
    def compute_padding(self, h_in, s, k_size):
        return max((h_in * (s - 2) - s + k_size) // 2, 0)

    def forward(self, z, y):
        z = z.reshape(self.batch_size, self.time_steps, self.n_noise_t)
        y = y[:, None, :].expand(self.batch_size, self.time_steps, self.n_noise_y)
        x = torch.cat([z, y], -1)
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm2(x)
        x = x.permute(0, 2, 1)
        if self.bn:
            x = self.lstmbn2(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(self.batch_size * self.time_steps, -1)
        x = self.dense_net(x)
        x = x.reshape(self.batch_size * self.time_steps, self.filter_size * 4, 8, 8)
        x = self.deconv_net(x)
        x = x.reshape(self.batch_size, self.time_steps, self.x_height, self.x_width)
        return x