"""
定义数据集类RemoteSensingImageDataset
"""
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from configs import data_config
from utils.data_utils import load_tiff_data, preprocess_data, load_mask_by_geo

class RemoteSensingImageDataset(Dataset):
    def __init__(self, data_paths_df, do_data_augment=False):
        self.data_paths = data_paths_df
        self.do_data_augment = do_data_augment

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 获取数据路径并加载
        matched_data_path = self.data_paths.iloc[idx]
        data_S2_target_path, data_MODIS_path, data_S1_path, \
            data_S2_ref_path, data_S2_before_path, data_S2_after_path = matched_data_path
        data_S2_target, geo_S2 = load_tiff_data(data_S2_target_path)
        data_MODIS, _ = load_tiff_data(data_MODIS_path)
        data_S1, _ = load_tiff_data(data_S1_path)
        data_S2_ref, _ = load_tiff_data(data_S2_ref_path)
        data_S2_before, _ = load_tiff_data(data_S2_before_path)
        data_S2_after, _ = load_tiff_data(data_S2_after_path)
        crop_mask = load_mask_by_geo(data_config.crop_mask_path, data_S2_target, geo_S2)

        # 预处理数据: 提取波段 标准化
        data_S2_target = preprocess_data(data_S2_target,
                                         data_config.S2_selected_bands,
                                         data_config.S2_max,
                                         data_config.S2_min,
                                         [0.5] * 8,
                                         [0.5] * 8)
        data_MODIS = preprocess_data(data_MODIS,
                                     data_config.MODIS_selected_bands,
                                     data_config.MODIS_max,
                                     data_config.MODIS_min,
                                     data_config.MODIS_mean,
                                     data_config.MODIS_std)
        data_S1 = preprocess_data(data_S1,
                                  data_config.S1_selected_bands,
                                  data_config.S1_max,
                                  data_config.S1_min,
                                  data_config.S1_mean,
                                  data_config.S1_std)
        data_S2_ref = preprocess_data(data_S2_ref,
                                      data_config.S2_selected_bands,
                                      data_config.S2_max,
                                      data_config.S2_min,
                                      data_config.S2_mean,
                                      data_config.S2_std)
        data_S2_before = preprocess_data(data_S2_before,
                                         data_config.S2_selected_bands,
                                         data_config.S2_max,
                                         data_config.S2_min,
                                         data_config.S2_mean,
                                         data_config.S2_std)
        data_S2_after = preprocess_data(data_S2_after,
                                        data_config.S2_selected_bands,
                                        data_config.S2_max,
                                        data_config.S2_min,
                                        data_config.S2_mean,
                                        data_config.S2_std)

        # 数据张量化
        data_S2_target = ToTensor()(data_S2_target)
        data_MODIS = ToTensor()(data_MODIS)
        data_S1 = ToTensor()(data_S1)
        data_S2_ref = ToTensor()(data_S2_ref)
        data_S2_before = ToTensor()(data_S2_before)
        data_S2_after = ToTensor()(data_S2_after)
        crop_mask = ToTensor()(crop_mask)

        # 数据增强
        if self.do_data_augment:
            # 随机水平翻转
            if torch.rand(1).item() > 0.5:
                data_S2_target = F.hflip(data_S2_target)
                data_MODIS = F.hflip(data_MODIS)
                data_S1 = F.hflip(data_S1)
                data_S2_ref = F.hflip(data_S2_ref)
                data_S2_before = F.hflip(data_S2_before)
                data_S2_after = F.hflip(data_S2_after)
                crop_mask = F.hflip(crop_mask)
            # 随机垂直翻转
            if torch.rand(1).item() > 0.5:
                data_S2_target = F.vflip(data_S2_target)
                data_MODIS = F.vflip(data_MODIS)
                data_S1 = F.vflip(data_S1)
                data_S2_ref = F.vflip(data_S2_ref)
                data_S2_before = F.vflip(data_S2_before)
                data_S2_after = F.vflip(data_S2_after)
                crop_mask = F.vflip(crop_mask)

        return {'S2_target': data_S2_target,
                'MODIS': data_MODIS,
                'S1': data_S1,
                'S2_reference': data_S2_ref,
                'S2_before': data_S2_before,
                'S2_after': data_S2_after,
                'crop_mask': crop_mask}


if __name__ == '__main__':
    pass
