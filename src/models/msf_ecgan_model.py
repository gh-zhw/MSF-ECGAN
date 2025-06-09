"""
定义模型类MSFECGANModel
"""
import os

import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary

from models.loss_functions import GANLoss, SAMLoss
from models.net_blocks import ResConvBlock, ResDeconvBlock


class MSFECGANModel:
    """
    Multi-source fusion enhanced CGAN
    """

    def __init__(self, opt):
        """
        Initialize the MSFECGANModel class
        :param opt: stores model flags
        """
        super().__init__()

        # recon_mode: 模型重建模式
        #   1: 双辅助数据+双参考数据
        #   2: MODIS辅助数据+双参考数据
        #   3: S1辅助数据+双参考数据
        #   4: 仅双参考数据
        #   5: 双辅助数据+单参考数据
        self.recon_mode = opt.recon_mode
        self.gan_mode = opt.gan_mode

        self.device = torch.device("cuda" if opt.use_gpu else "cpu")
        self.net_G = self.build_G()
        self.net_D = self.build_D()

        if opt.is_train:
            # 损失函数
            self.criterionGAN = GANLoss(opt.gan_mode, self.device)
            self.criterionL1 = nn.L1Loss()
            self.with_sam_loss = opt.with_sam_loss
            if opt.with_sam_loss:
                self.criterionSAM = SAMLoss(opt.sam_unit)
            # 损失系数
            self.lambda_L1 = opt.lambda_l1
            self.lambda_SAM = opt.lambda_sam

            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
        else:
            self.load_model(opt.test_model_path)

    def set_input(self, input_dict: dict):
        """
        获取输入数据
        :param input_dict: 输入数据字典 optional_keys: ['S2_target', 'MODIS', 'S1', 'S2_reference', 'S2_before', 'S2_after']
        :return: S2_target, MODIS_input, S1_input, S2_ref_1_input, S2_ref_2_input
        """
        if 'S2_target' in input_dict.keys():
            S2_target = input_dict['S2_target']
        else:
            S2_target = torch.empty(1)

        if self.recon_mode == 1:
            MODIS_input = input_dict['MODIS']
            S1_input = input_dict['S1']
            S2_ref_1_input = input_dict['S2_before']
            S2_ref_2_input = input_dict['S2_after']
        elif self.recon_mode == 2:
            MODIS_input = input_dict['MODIS']
            S1_input = torch.empty(1)
            S2_ref_1_input = input_dict['S2_before']
            S2_ref_2_input = input_dict['S2_after']
        elif self.recon_mode == 3:
            MODIS_input = torch.empty(1)
            S1_input = input_dict['S1']
            S2_ref_1_input = input_dict['S2_before']
            S2_ref_2_input = input_dict['S2_after']
        elif self.recon_mode == 4:
            MODIS_input = torch.empty(1)
            S1_input = torch.empty(1)
            S2_ref_1_input = input_dict['S2_before']
            S2_ref_2_input = input_dict['S2_after']
        elif self.recon_mode == 5:
            MODIS_input = input_dict['MODIS']
            S1_input = input_dict['S1']
            S2_ref_1_input = input_dict['S2_reference']
            S2_ref_2_input = torch.empty(1)

        return S2_target.to(self.device), MODIS_input.to(self.device), S1_input.to(self.device), \
            S2_ref_1_input.to(self.device), S2_ref_2_input.to(self.device)

    def build_G(self):
        """"
        build G
        :return: G object
        """
        if self.recon_mode == 1:
            input_nc1, input_nc2 = 18, 6
        elif self.recon_mode == 2:
            input_nc1, input_nc2 = 16, 6
        elif self.recon_mode == 3:
            input_nc1, input_nc2 = 18, 0
        elif self.recon_mode == 4:
            input_nc1, input_nc2 = 16, 0
        elif self.recon_mode == 5:
            input_nc1, input_nc2 = 10, 6

        return MSFECGANGenerator(input_nc1, input_nc2).to(self.device)

    def build_D(self):
        """
        build D
        :return: D object
        """
        if self.recon_mode in [1, 3, 5]:
            input_nc = 10
        elif self.recon_mode in [2, 4]:
            input_nc = 8

        if self.gan_mode == 'vanilla':
            output_sig = True
        elif self.gan_mode in ['lsgan', 'wgangp']:
            output_sig = False

        return MSFECGANDiscriminator(input_nc, output_sig).to(self.device)

    def optimize_G(self, S2_target, MODIS_input, S1_input, S2_ref_1_input, S2_ref_2_input):
        """优化生成器参数"""
        # G forward
        G_output = self.net_G(MODIS_input, S1_input, S2_ref_1_input, S2_ref_2_input)
        # D forward
        D_fake_input = G_output
        if S1_input.numel() > 0:
            D_fake_input = torch.cat((D_fake_input, S1_input), dim=1)
        D_output = self.net_D(D_fake_input)
        # Calculate G loss
        loss_G_GAN = self.criterionGAN(D_output, target_is_real=True)
        loss_G_L1 = self.criterionL1(G_output, S2_target)
        if self.with_sam_loss:
            loss_G_SAM = self.criterionSAM(G_output, S2_target)
            loss_G = loss_G_GAN + self.lambda_L1 * loss_G_L1 + self.lambda_SAM * loss_G_SAM
        else:
            loss_G = loss_G_GAN + self.lambda_L1 * loss_G_L1
        # G backward
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        loss_G_dict = {'loss_G_GAN': loss_G_GAN.item(),
                       'loss_G_L1': loss_G_L1.item()}
        if self.with_sam_loss:
            loss_G_dict['loss_G_SAM'] = loss_G_SAM.item()

        return loss_G_dict, G_output

    def optimize_D(self, G_output, S2_target, S1_input):
        """优化判别器参数"""
        # D forward
        D_fake_input = G_output.detach()
        D_real_input = S2_target
        if S1_input.numel() > 0:
            D_fake_input = torch.cat((D_fake_input, S1_input), dim=1)
            D_real_input = torch.cat((D_real_input, S1_input), dim=1)
        D_fake_output = self.net_D(D_fake_input)
        D_real_output = self.net_D(D_real_input)
        # # Calculate D loss
        loss_D_fake = self.criterionGAN(D_fake_output, target_is_real=False)
        loss_D_real = self.criterionGAN(D_real_output, target_is_real=True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        # G backward
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()

        return loss_D

    def optimize_model(self, input_dict: dict):
        """
        优化模型参数
        :param input_dict: 输入数据字典 optional_keys: ['S2_target', 'MODIS', 'S1', 'S2_reference', 'S2_before', 'S2_after']
        :return: loss dict (keys: ['loss_D_GAN', 'loss_G_GAN', 'loss_G_L1', 'loss_G_SAM'])
        """
        S2_target, MODIS_input, S1_input, S2_ref_1_input, S2_ref_2_input = self.set_input(input_dict)
        loss_G_dict, G_output = self.optimize_G(S2_target,
                                                MODIS_input,
                                                S1_input,
                                                S2_ref_1_input,
                                                S2_ref_2_input)
        loss_D = self.optimize_D(G_output, S2_target, S1_input)

        loss_dict = dict()
        loss_dict['loss_D_GAN'] = loss_D.item()
        loss_dict.update(loss_G_dict)
        return loss_dict

    def predict(self, input_dict: dict):
        """
        生成器进行预测生成
        :param input_dict: 输入数据字典
        :return: 生成结果
        """
        S2_target, MODIS_input, S1_input, S2_ref_1_input, S2_ref_2_input = self.set_input(input_dict)

        self.eval()
        with torch.no_grad():
            prediction = self.net_G(MODIS_input, S1_input, S2_ref_1_input, S2_ref_2_input)

        self.train()
        return prediction

    def save_model(self, saving_dir: str, epoch, metric_dict: dict = None):
        """
        保存模型
        :param saving_dir: 模型保存文件夹路径
        :param epoch: 模型保存时的训练轮次 or 'best'
        :param metric_dict: 模型各项指标 metric_keys: ['MAE', 'MSE', 'PSNR', 'SAM', 'SSIM']
        :return: None
        """
        if epoch == 'best':
            model_G_name = "best_MSFECGAN_G"
            model_D_name = "best_MSFECGAN_D"
        else:
            loss_description = "Pix2Pix"
            if self.with_sam_loss:
                loss_description += "_SAM"

            model_G_name = f"MSFECGAN_G_{loss_description}_mode_{self.recon_mode}_epoch_{epoch}"
            if metric_dict:
                model_G_name += f"_MAE_{metric_dict['MAE']:.4f}_SAM_{metric_dict['SAM']:.4f}_SSIM_{metric_dict['SSIM']:.4f}"
            model_D_name = f"MSFECGAN_D_{loss_description}_mode_{self.recon_mode}_epoch_{epoch}"

        path_G_model = os.path.join(saving_dir, model_G_name)
        path_D_model = os.path.join(saving_dir, model_D_name)

        torch.save(self.net_G.state_dict(), path_G_model)
        print(f"Model.Net_G successfully saved to '{path_G_model}'.")
        torch.save(self.net_D.state_dict(), path_D_model)
        print(f"Model.Net_D successfully saved to '{path_D_model}'.")

    def load_model(self, path_G_model: str, path_D_model: str = ""):
        """
        加载模型权重
        :param path_G_model: 生成器权重路径
        :param path_D_model: 判别器权重路径 (optional)
        :return: None
        """
        self.net_G.load_state_dict(torch.load(path_G_model))
        print(f"Model.Net_G successfully loaded from {path_G_model}.")

        if path_D_model != "":
            self.net_D.load_state_dict(torch.load(path_D_model))
            print(f"Model.Net_D successfully loaded from {path_D_model}.")

    def train(self):
        """
        将模型设置为训练状态
        :return: None
        """
        self.net_G.train()
        self.net_D.train()

    def eval(self):
        """
        将模型设置为验证状态
        :return: None
        """
        self.net_G.eval()
        self.net_D.eval()


class MSFECGANGenerator(nn.Module):
    def __init__(self, input_nc1: int, input_nc2: int, conv_norm='BN', deconv_norm='IN'):
        """
        定义生成器的结构
        :param input_nc1: SAR data & Ref data 的输入通道
        :param input_nc2: MODIS data 的输入通道
        :param conv_norm: 卷积块的归一化方式
        :param deconv_norm: 反卷积块的归一化方式
        """
        super(MSFECGANGenerator, self).__init__()

        # 定义ResConvBlock参数
        res_conv_channels = [input_nc1, 36, 72, 144, 288, 576, 1152]
        # (kernel_size, stride, padding)
        res_conv_params = [(3, 2, 1), (3, 2, 2), (3, 2, 1), (2, 2, 0), (2, 2, 0), (2, 2, 1)]

        # 定义ResDeconvBlock参数
        res_deconv_channels = [
            (1152 + input_nc2, 579), (1155, 578), (866, 433), (577, 288), (360, 90), (126, 32)
        ]
        # (kernel_size, stride, padding)
        res_deconv_params = [(4, 1, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0), (3, 2, 2), (2, 2, 0)]

        # 构建ResConvBlocks
        self.res_conv_blocks = nn.ModuleList([
            ResConvBlock(res_conv_channels[i],
                         res_conv_channels[i + 1],
                         kernel_size=k,
                         stride=s,
                         padding=p,
                         norm=conv_norm)
            for i, (k, s, p) in enumerate(res_conv_params)
        ])

        # 构建ResDeconvBlocks
        self.res_deconv_blocks = nn.ModuleList([
            ResDeconvBlock(res_deconv_channels[i][0],
                           res_deconv_channels[i][1],
                           kernel_size=k,
                           stride=s,
                           padding=p,
                           norm=deconv_norm)
            for i, (k, s, p) in enumerate(res_deconv_params)
        ])

        if input_nc2:
            self.input_MODIS_conv_block = ResConvBlock(in_channel=6,
                                                       out_channel=6,
                                                       kernel_size=3,
                                                       stride=1,
                                                       padding=1,
                                                       norm=conv_norm)
        self.output_conv_block = nn.Sequential(
            ResConvBlock(32 + input_nc1, 24, kernel_size=3, stride=1, padding=1, norm=deconv_norm),
            ResConvBlock(24, 16, kernel_size=3, stride=1, padding=1, norm=deconv_norm),
            ResConvBlock(16, 8, kernel_size=3, stride=1, padding=1, norm=deconv_norm),
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, MODIS_input, S1_input, S2_ref_1_input, S2_ref_2_input):
        concat_input = S2_ref_1_input  # 至少输入单个参考数据
        if S1_input.numel() > 0:
            concat_input = torch.cat((S1_input, concat_input), dim=1)
        if S2_ref_2_input.numel() > 0:
            concat_input = torch.cat((S2_ref_2_input, concat_input), dim=1)

        # Forward through ResConvBlocks
        S1_ref_features = []
        x = concat_input
        for res_conv_block in self.res_conv_blocks:
            x = res_conv_block(x)
            S1_ref_features.append(x)

        # 有无 MODIS 输入 不同处理
        if MODIS_input.numel() > 0:
            MODIS_feature = self.input_MODIS_conv_block(MODIS_input)
            x = MODIS_feature
        else:
            x = None

        # Forward through ResDeconvBlocks
        for i, res_deconv_block in enumerate(self.res_deconv_blocks):
            if x is not None:
                x = torch.cat((x, S1_ref_features[-(i + 1)]), dim=1)
            else:
                x = S1_ref_features[-(i + 1)]
            x = res_deconv_block(x)

        output = self.output_conv_block(torch.cat((x, concat_input), dim=1))
        return nn.Tanh()(output)


class MSFECGANDiscriminator(nn.Module):
    def __init__(self, input_nc: int, output_sig: bool):
        """
        定义判别器的结构
        :param input_nc: 输入通道
        :param output_sig: 输出是否经过sigmoid函数
        """
        super(MSFECGANDiscriminator, self).__init__()

        conv_channels = [input_nc, 20, 40, 80, 40, 20, 10, 5]
        # (kernel_size, stride, padding)
        conv_params = [(3, 2, 1), (3, 2, 1), (3, 2, 1), (2, 2, 0), (2, 2, 0), (2, 2, 0), (2, 2, 0)]

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(conv_channels[i], conv_channels[i + 1], kernel_size=k, stride=s, padding=p),
                nn.BatchNorm2d(conv_channels[i + 1]),
                nn.LeakyReLU(inplace=True)
            )
            for i, (k, s, p) in enumerate(conv_params)
        ])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * 2 * 2, 1),
        )

        if output_sig:
            self.fc.add_module("sigmoid", nn.Sigmoid())

    def forward(self, input):
        x = input
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        output = self.fc(x)
        return output


if __name__ == '__main__':
    # MODIS & S1 & S2_before & S2_after
    g = MSFECGANGenerator(input_nc1=18, input_nc2=6, conv_norm='BN', deconv_norm='IN')
    summary(g.cuda(), [(6, 5, 5), (2, 250, 250), (8, 250, 250), (8, 250, 250)], 4)
    d = MSFECGANDiscriminator(input_nc=10, output_sig=False)
    summary(d.cuda(), [(10, 250, 250)], 4)

    # MODIS & S2_before & S2_after
    # g = BasicGenerator(input_nc1=16, input_nc2=6, conv_norm='BN', deconv_norm='IN')
    # summary(g.cuda(), [(6, 5, 5), (0, 0, 0), (8, 250, 250), (8, 250, 250)], 4)
    # d = BasicDiscriminator(input_nc=8, output_sig=False)
    # summary(d.cuda(), [(8, 250, 250)], 4)

    # S1 & S2_before & S2_after
    # g = BasicGenerator(input_nc1=18, input_nc2=0, conv_norm='BN', deconv_norm='IN')
    # summary(g.cuda(), [(0, 0, 0), (2, 250, 250), (8, 250, 250), (8, 250, 250)], 4)
    # d = BasicDiscriminator(input_nc=10, output_sig=False)
    # summary(d.cuda(), [(10, 250, 250)], 4)

    # S2_before & S2_after
    # g = BasicGenerator(input_nc1=16, input_nc2=0, conv_norm='BN', deconv_norm='IN')
    # summary(g.cuda(), [(0, 0, 0), (0, 0, 0), (8, 250, 250), (8, 250, 250)], 4)
    # d = BasicDiscriminator(input_nc=8, output_sig=False)
    # summary(d.cuda(), [(8, 250, 250)], 4)

    # MODIS & S1 & S2_ref
    # g = BasicGenerator(input_nc1=10, input_nc2=6, conv_norm='BN', deconv_norm='IN')
    # summary(g.cuda(), [(6, 5, 5), (2, 250, 250), (8, 250, 250), (0, 0, 0)], 4)
    # d = BasicDiscriminator(input_nc=10, output_sig=False)
    # summary(d.cuda(), [(10, 250, 250)], 4)
