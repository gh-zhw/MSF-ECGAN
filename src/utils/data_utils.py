"""
导入和处理数据的工具
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal

from configs import data_config


def load_tiff_data(data_path: str, use_cache=False, cache_root=data_config.npy_cache_path):
    """
    加载tiff数据，支持缓存机制。
    :param data_path: 原始.tif数据路径
    :param use_cache: 是否使用缓存（默认为 False）
    :param cache_root: 缓存根目录
    :return: npy格式数据 HWC , 地理信息
    """
    if use_cache:
        rel_path = os.path.relpath(data_path, data_config.data_root_directory)
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        npy_path = os.path.join(cache_root, rel_path_no_ext + '.npy')
        geo_path = os.path.join(cache_root, rel_path_no_ext + '.geo.npy')

        if os.path.exists(npy_path) and os.path.exists(geo_path):
            npy_data = np.load(npy_path)
            geo_info = np.load(geo_path, allow_pickle=True).item()
            return npy_data, (geo_info['geo_transform'], geo_info['projection'])

    # 正常读取 tiff
    tiff_data = gdal.Open(data_path, gdal.GA_ReadOnly)
    if tiff_data is None:
        raise FileNotFoundError(f"无法打开文件: {data_path}")

    geo_transform = tiff_data.GetGeoTransform()
    projection = tiff_data.GetProjection()

    num_bands = tiff_data.RasterCount
    width = tiff_data.RasterXSize
    height = tiff_data.RasterYSize

    npy_data = np.zeros((height, width, num_bands), dtype=np.int16)
    for i in range(1, num_bands + 1):
        band_data = tiff_data.GetRasterBand(i).ReadAsArray()
        npy_data[:, :, i - 1] = band_data

    # 若启用缓存，则保存
    if use_cache:
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, npy_data)
        np.save(geo_path, {'geo_transform': geo_transform, 'projection': projection})

    return npy_data, (geo_transform, projection)


def preprocess_data(data, selected_bands, max_val, min_val, data_mean, data_std):
    """
    预处理加载tiff数据得到的Numpy数据
    :param data: Numpy数据
    :param selected_bands: 所选波段
    :param max_val: 数据可取的最大值
    :param min_val: 数据可取的最小值
    :param data_mean: 训练数据归一化后的平均值
    :param data_std: 训练数据归一化后的标准差
    :return: 预处理后的Numpy数据
    """
    # 提取所选波段
    processed_data = data[:, :, selected_bands].astype(np.float32)
    # 有效范围截断
    np.clip(processed_data, min_val, max_val, out=processed_data)
    # 归一化
    processed_data -= min_val
    processed_data /= (max_val - min_val)
    # 标准化
    processed_data -= data_mean
    processed_data /= data_std

    return processed_data

def load_date_match_xlsx(xlsx_path: str):
    """
    加载时间匹配文件并返回对应数据表
    :param xlsx_path: 时间匹配文件路径
    :return: 训练数据匹配表, 测试数据匹配表
    """
    date_match_xlsx = pd.ExcelFile(xlsx_path)
    sheet_names = date_match_xlsx.sheet_names

    data_keys = ["S2_target", "MODIS", "S1", "S2_reference", "S2_before", "S2_after"]

    # 获取训练数据日期
    train_date_match_df = pd.DataFrame(columns=data_keys)
    for sheet_name in sheet_names[:-1]:
        sheet_data = pd.read_excel(xlsx_path, sheet_name=sheet_name,
                                   usecols=data_keys, dtype="str")
        train_date_match_df = pd.concat([train_date_match_df, sheet_data],
                                        ignore_index=True)

    # 获取测试数据日期
    test_date_match_df = pd.read_excel(xlsx_path, sheet_name=sheet_names[-1],
                                       usecols=data_keys, dtype="str")

    return train_date_match_df, test_date_match_df

def get_matched_data_paths(date_match_df, is_train_data: bool):
    """
    根据时间匹配表获取数据路径
    :param date_match_df: 时间匹配表
    :param is_train_data: 是否为训练数据
    :return: 数据路径表
    """
    # 定义常量
    if is_train_data:
        patch_number_of_roi = [12, 18, 8, 12, 6, 8, 6, 10, 6, 10]
        date_num_of_roi = [32, 31, 16, 35, 32, 12, 31, 23, 11, 36]
        rois = ['NB_roi_1', 'NB_roi_2', 'NB_roi_3', 'NB_roi_4', 'NB_roi_5',
                'NB_roi_6', 'NB_roi_7', 'NB_roi_8', 'NB_roi_9', 'NB_roi_10']
        purpose = 'train'
    else:
        patch_number_of_roi = [21]
        date_num_of_roi = [11]
        rois = ['NB_roi_test']
        purpose = 'test'

    roi_index = 0
    data_keys = ["S2_target", "MODIS", "S1", "S2_reference", "S2_before", "S2_after"]
    # 创建数据路径表
    data_paths_df = pd.DataFrame(columns=data_keys)
    # 按行遍历所有匹配时间
    for date_index, dates in date_match_df.iterrows():
        roi = rois[roi_index]
        date_S2_target = dates["S2_target"]
        date_MODIS = dates["MODIS"]
        date_S1 = dates["S1"]
        date_S2_ref = dates["S2_reference"]
        date_S2_before = dates["S2_before"]
        date_S2_after = dates["S2_after"]
        # 遍历当前roi的每个patch
        for patch_index in range(1, patch_number_of_roi[roi_index] + 1):
            # S2_target
            data_S2_target_path = f"{purpose}/S2/{roi}_{date_S2_target}/S2_{roi}_{date_S2_target}_{patch_index:03}.tif"
            data_S2_target_path = os.path.join(data_config.data_root_directory, data_S2_target_path)
            # MODIS
            data_MODIS_path = f"{purpose}/MODIS/{roi}_{date_MODIS}/MODIS_{roi}_{date_MODIS}_{patch_index:03}.tif"
            data_MODIS_path = os.path.join(data_config.data_root_directory, data_MODIS_path)
            # S1
            data_S1_path = f"{purpose}/S1/{roi}_{date_S1}/S1_{roi}_{date_S1}_{patch_index:03}.tif"
            data_S1_path = os.path.join(data_config.data_root_directory, data_S1_path)
            # S2_reference
            data_S2_ref_path = f"{purpose}/S2/{roi}_{date_S2_ref}/S2_{roi}_{date_S2_ref}_{patch_index:03}.tif"
            data_S2_ref_path = os.path.join(data_config.data_root_directory, data_S2_ref_path)
            # S2_before
            data_S2_before_path = f"{purpose}/S2/{roi}_{date_S2_before}/S2_{roi}_{date_S2_before}_{patch_index:03}.tif"
            data_S2_before_path = os.path.join(data_config.data_root_directory, data_S2_before_path)
            # S2_after
            data_S2_after_path = f"{purpose}/S2/{roi}_{date_S2_after}/S2_{roi}_{date_S2_after}_{patch_index:03}.tif"
            data_S2_after_path = os.path.join(data_config.data_root_directory, data_S2_after_path)

            # 添加到数据路径表中
            matched_data_path = pd.Series([data_S2_target_path, data_MODIS_path, data_S1_path,
                                           data_S2_ref_path, data_S2_before_path, data_S2_after_path],
                                          index=data_keys)
            data_paths_df = pd.concat([data_paths_df, matched_data_path.to_frame().T], ignore_index=True)

        # 当前roi的数据已保存完 到下一个roi
        if date_index == sum(date_num_of_roi[:roi_index + 1]) - 1:
            roi_index += 1

    return data_paths_df

def load_mask_by_geo(mask_path: str, optical_data, optical_geo):
    """
    根据光学影像的地理信息，从掩膜文件中读取对应区域的数据（避免整图加载）
    :param mask_path: 掩膜tif路径
    :param mask_geo: 掩膜图像的地理信息（GeoTransform, Projection）
    :param optical_data: 光学影像数据（np.array）
    :param optical_geo: 光学影像的地理信息（GeoTransform, Projection）
    :return: 裁剪出的掩膜数组（np.array）
    """
    mask_tiff = gdal.Open(mask_path, gdal.GA_ReadOnly)
    mask_transform = mask_tiff.GetGeoTransform()

    optical_array, (optical_transform, _) = optical_data, optical_geo

    # 获取像素分辨率
    mask_origin_x, mask_pixel_width, _, mask_origin_y, _, mask_pixel_height = mask_transform
    optical_origin_x, optical_pixel_width, _, optical_origin_y, _, optical_pixel_height = optical_transform

    # 确保分辨率一致
    assert abs(mask_pixel_width - optical_pixel_width) < 1e-6, "像元大小不同"
    assert abs(mask_pixel_height - optical_pixel_height) < 1e-6, "像元大小不同"

    # 获取光学影像的尺寸
    optical_height, optical_width = optical_array.shape[:2]

    # 计算在mask中的偏移
    offset_x = int((optical_origin_x - mask_origin_x) / mask_pixel_width)
    offset_y = int((optical_origin_y - mask_origin_y) / mask_pixel_height)

    # 使用GDAL按窗口读取对应区域的mask
    mask_band = mask_tiff.GetRasterBand(1)
    mask_data = mask_band.ReadAsArray(offset_x, offset_y, optical_width, optical_height)

    return mask_data.astype(np.int16)


if __name__ == "__main__":
    train_date_match, test_date_match = load_date_match_xlsx(data_config.date_match_xlsx_path)
    # print(train_date_match.info(verbose=True))
    # print(test_date_match.info(verbose=True))

    train_data_paths = get_matched_data_paths(train_date_match, is_train_data=True)
    # print(train_data_paths.info(verbose=True))

    test_data_paths = get_matched_data_paths(test_date_match, is_train_data=False)
    # print(test_data_paths.info(verbose=True))

    data_S2_path = train_data_paths["S2_target"].iloc[0]
    npy_data_S2, geo_info = load_tiff_data(data_S2_path)

    processed_data_S2 = preprocess_data(npy_data_S2,
                                        data_config.S2_selected_bands,
                                        data_config.S2_max,
                                        data_config.S2_min,
                                        data_config.S2_mean,
                                        data_config.S2_std)
    print(processed_data_S2.shape, processed_data_S2.dtype,
          processed_data_S2.min(), processed_data_S2.max())

