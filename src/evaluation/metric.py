"""
定义指标计算函数
"""
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calc_sam(prediction, target, epsilon=1e-7):
    """
    计算 prediction 与 target 的光谱角 SAM
    :param prediction: 预测图像
    :param target: 真实图像
    :param epsilon: 避免数值计算中的除零或精度问题
    :return: 光谱角 SAM
    """
    dot_product = torch.mul(prediction, target).sum(dim=1)
    norm1 = torch.linalg.norm(prediction, dim=1)
    norm2 = torch.linalg.norm(target, dim=1)

    cos_similarity = dot_product / (norm1 * norm2 + epsilon)
    sam = torch.acos(torch.clamp(cos_similarity, -1.0 + epsilon, 1.0 - epsilon))
    mean_sam = sam.mean()

    return mean_sam


def calc_metrics(prediction, target, denorm: bool = False, max_val=1, data_range=1):
    """
    计算 prediction 与 target 的各项指标
    :param denorm: 是否对输入进行反标准化
    :param prediction: 预测图像
    :param target: 真实图像
    :param max_val: 最大像素值
    :param data_range: 像素值的范围
    :return: 各项指标字典 metric_keys: ['MAE', 'MSE', 'PSNR', 'SAM', 'SSIM']
    """
    # 保证数据在同一个device
    if target.device != prediction.device:
        target = target.to(prediction.device)

    if denorm:
        # 反标准化至(0, 1)
        prediction = (prediction + 1) / 2
        target = (target + 1) / 2

    # MAE
    mae = F.l1_loss(prediction, target)
    # MSE
    mse = F.mse_loss(prediction, target)
    # SAM
    sam = calc_sam(prediction, target)
    # PSNR
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    # SSIM
    ssim_val = ssim(prediction, target, data_range=data_range)

    return {'MAE': mae.item(),
            'MSE': mse.item(),
            'PSNR': psnr.item(),
            'SAM': sam.item(),
            'SSIM': ssim_val.item()}


if __name__ == '__main__':
    # (B, C, W, H)
    prediction = torch.rand((8, 8, 250, 250), dtype=torch.float32) * 2 - 1
    target = torch.rand((8, 8, 250, 250), dtype=torch.float32) * 2 - 1

    metric_dict = calc_metrics(prediction, target, denorm=True)
    for key, val in metric_dict.items():
        print(f"{key} = {val}")
