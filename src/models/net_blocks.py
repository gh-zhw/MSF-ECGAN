import torch.nn as nn


class BasicConvBlock(nn.Module):
    """
    基础卷积块
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, norm_choice='BN'):
        super(BasicConvBlock, self).__init__()

        self.norm_choice = norm_choice
        self.conv_1 = nn.Conv2d(in_channel,
                                out_channel,
                                kernel_size,
                                stride,
                                padding,
                                bias=False)
        self.norms = nn.ModuleDict({'IN': nn.InstanceNorm2d(out_channel),
                                    'BN': nn.BatchNorm2d(out_channel)})
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.norms[self.norm_choice](out)
        out = self.activation(out)
        out = self.conv_2(out)
        return out


class BasicDeconvBlock(nn.Module):
    """
    基础反卷积块
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, norm_choice='BN'):
        super(BasicDeconvBlock, self).__init__()

        self.norm_choice = norm_choice

        self.deconv = nn.ConvTranspose2d(in_channel,
                                         out_channel,
                                         kernel_size,
                                         stride,
                                         padding,
                                         bias=False)
        self.norms = nn.ModuleDict({'IN': nn.InstanceNorm2d(out_channel),
                                    'BN': nn.BatchNorm2d(out_channel)})
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.deconv(x)
        out = self.norms[self.norm_choice](out)
        out = self.activation(out)
        out = self.conv(out)
        return out


class ResConvBlock(nn.Module):
    """
    带残差连接的卷积块
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, norm='BN'):
        super().__init__()

        self.conv = BasicConvBlock(in_channel,
                                   out_channel,
                                   kernel_size,
                                   stride,
                                   padding,
                                   norm_choice=norm)
        self.skip_connect = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out + self.skip_connect(x))
        return out


class ResDeconvBlock(nn.Module):
    """
    带残差连接的反卷积块
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, norm='BN'):
        super().__init__()

        self.deconv = BasicDeconvBlock(in_channel,
                                       out_channel,
                                       kernel_size,
                                       stride,
                                       padding,
                                       norm_choice=norm)
        self.skip_connect = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.deconv(x)
        out = self.activation(out + self.skip_connect(x))
        return out


if __name__ == '__main__':
    pass
