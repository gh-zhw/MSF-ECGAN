"""
定义损失函数类
"""
import torch
from torch import nn

from evaluation.metric import calc_sam


class GANLoss:
    def __init__(self, gan_mode: str, device, target_real_label=1.0, target_fake_label=0.0):
        """
        Initialize the GANLoss class
        :param gan_mode: the type of GANLoss objective
        :param target_real_label: label for a real image
        :param target_fake_label: label of a fake image
        """
        self.real_label = torch.tensor(target_real_label).to(device)
        self.fake_label = torch.tensor(target_fake_label).to(device)

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('Gan loss mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real: bool):
        """
        Create label tensors with the same size as the input
        :param prediction: typically the prediction from a discriminator
        :param target_is_real: if the ground truth label is for real images or fake images
        :return: A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        Calculate loss given Discriminator's output and ground truth labels
        :param prediction: typically the prediction from a discriminator
        :param target_is_real: if the ground truth label is for real images or fake images
        :return: the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class SAMLoss(nn.Module):
    def __init__(self, unit_mode: str = 'rad', eps=1e-7):
        """
        Initialize the SAMLoss class
        :param unit_mode: SAM 的单位 弧度制 'rad' 或 角度制 'deg'
        :param eps: 避免数值计算中的除零或精度问题
        """
        super(SAMLoss, self).__init__()
        self.unit_mode = unit_mode
        self.eps = eps

    def forward(self, prediction, target):
        sam_mean = calc_sam(prediction, target, self.eps)

        if self.unit_mode == 'deg':
            sam_mean = torch.rad2deg(sam_mean)
        return sam_mean


def calc_adaptive_weights(rice_mask, eps=1e-6, max_weight=10.0):
    """
    根据水稻掩膜计算自适应权重，归一化后使权重和为2。
    
    输入:
        rice_mask: Tensor (B, 1, H, W)，水稻区域掩膜
        eps: 避免除零的小常数
        max_weight: 权重上限裁剪，防止极端权重
    
    返回:
        lambda_rice: Tensor (B,) 加权水稻区域权重
        lambda_nonrice: Tensor (B,) 加权非水稻区域权重
    """
    rice_mask = rice_mask.float()
    nonrice_mask = 1.0 - rice_mask

    rice_area = rice_mask.sum(dim=[1, 2, 3])
    nonrice_area = nonrice_mask.sum(dim=[1, 2, 3])
    total_area = rice_area + nonrice_area

    lambda_rice = total_area / (rice_area + eps)
    lambda_nonrice = total_area / (nonrice_area + eps)

    lambda_rice = torch.where(rice_area > eps, lambda_rice, torch.zeros_like(lambda_rice))
    lambda_nonrice = torch.where(nonrice_area > eps, lambda_nonrice, torch.zeros_like(lambda_nonrice))

    lambda_rice = torch.clamp(lambda_rice, max=max_weight)
    lambda_nonrice = torch.clamp(lambda_nonrice, max=max_weight)

    weight_sum = lambda_rice + lambda_nonrice + eps
    lambda_rice = lambda_rice / weight_sum
    lambda_nonrice = lambda_nonrice / weight_sum

    return lambda_rice, lambda_nonrice


class AdaptiveWeightedL1Loss(nn.Module):
    def __init__(self, eps=1e-6, max_weight=10.0):
        super().__init__()
        self.eps = eps
        self.max_weight = max_weight

    def forward(self, prediction, target, rice_mask):
        abs_error = torch.abs(prediction - target)
        lambda_rice, lambda_nonrice = calc_adaptive_weights(rice_mask, self.eps, self.max_weight)

        rice_loss = (abs_error * rice_mask).mean(dim=[1, 2, 3])
        nonrice_loss = (abs_error * (1.0 - rice_mask)).mean(dim=[1, 2, 3])

        total_loss = lambda_rice * rice_loss + lambda_nonrice * nonrice_loss
        return total_loss.mean()


class AdaptiveWeightedSAMLoss(nn.Module):
    def __init__(self, unit_mode='rad', eps=1e-6, max_weight=10.0):
        super().__init__()
        self.unit_mode = unit_mode
        self.eps = eps
        self.max_weight = max_weight

    def forward(self, prediction, target, rice_mask):
        B, C, H, W = prediction.shape
        pred_flat = prediction.view(B, C, -1)
        target_flat = target.view(B, C, -1)

        lambda_rice, lambda_nonrice = calc_adaptive_weights(rice_mask, self.eps, self.max_weight)

        dot = (pred_flat * target_flat).sum(dim=1)
        pred_norm = pred_flat.norm(dim=1).clamp(min=self.eps)
        target_norm = target_flat.norm(dim=1).clamp(min=self.eps)

        cos_sim = dot / (pred_norm * target_norm)
        cos_sim = torch.clamp(cos_sim, min=-1 + 1e-7, max=1 - 1e-7)

        angle = torch.acos(cos_sim)
        if self.unit_mode == 'deg':
            angle = torch.rad2deg(angle)

        angle = angle.view(B, 1, H, W)

        rice_loss = (angle * rice_mask).mean(dim=[1, 2, 3])
        nonrice_loss = (angle * (1.0 - rice_mask)).mean(dim=[1, 2, 3])

        total_loss = lambda_rice * rice_loss + lambda_nonrice * nonrice_loss
        return total_loss.mean()


if __name__ == '__main__':
    pass
