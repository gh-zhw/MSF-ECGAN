"""
加载数据集
"""
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from configs import data_config
from dataset.define_dataset import RemoteSensingImageDataset
from utils.data_utils import load_date_match_xlsx, get_matched_data_paths


def create_dataset():
    """
    创建数据集
    :return: 训练数据集, 验证数据集, 测试数据集
    """
    # 加载数据路径
    train_date_match, test_date_match = load_date_match_xlsx(data_config.date_match_xlsx_path)
    train_data_paths = get_matched_data_paths(train_date_match, is_train_data=True)
    test_data_paths = get_matched_data_paths(test_date_match, is_train_data=False)

    # 将数据集分割成训练集和验证集
    train_data_paths, valid_data_paths = train_test_split(train_data_paths,
                                                          test_size=0.1,
                                                          random_state=0)

    train_dataset = RemoteSensingImageDataset(train_data_paths, do_data_augment=True)
    valid_dataset = RemoteSensingImageDataset(valid_data_paths, do_data_augment=False)
    test_dataset = RemoteSensingImageDataset(test_data_paths, do_data_augment=False)

    return train_dataset, valid_dataset, test_dataset


def get_dataloader(batch_size, train_dataset, valid_dataset, test_dataset, num_workers=0):
    """
    获取数据加载器DataLoader
    :param batch_size: 批大小
    :param train_dataset: 训练数据集
    :param valid_dataset: 验证数据集
    :param test_dataset: 测试数据集
    :param num_workers: 加载数据的子进程数
    :return: 训练数据加载器, 验证数据加载器, 测试数据加载器
    """
    train_dataloader = DataLoader(train_dataset,
                                  batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=num_workers)

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataset, valid_dataset, test_dataset = create_dataset()
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    BATCH_SIZE = 4
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(BATCH_SIZE,
                                                                         train_dataset,
                                                                         valid_dataset,
                                                                         test_dataset)
    print(len(train_dataloader), len(valid_dataloader), len(test_dataloader))
    print(train_dataloader.dataset[0]["crop_mask"].shape)