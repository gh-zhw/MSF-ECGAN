"""
测试命令行参数
"""

import argparse
from configs import test_config


def get_test_opt():
    parser = argparse.ArgumentParser(description="Model Testing")

    parser.add_argument("--batch_size", type=int, default=test_config.BATCH_SIZE,
                        help="Size of each batch")

    parser.add_argument("--recon_mode", type=int, default=1,
                        help="Reconstruction mode "
                             "(1: Double auxiliary data + double reference data, "
                             "2: MODIS auxiliary data + double reference data, "
                             "3: S1 auxiliary data + double reference data, "
                             "4: Only double reference data, "
                             "5: Double auxiliary data + single reference data)")
    parser.add_argument("--gan_mode", type=str, default='lsgan',
                        help="Type of GAN loss (options: 'vanilla', 'lsgan', 'wgangp')")

    parser.add_argument("--test_model_path", type=str, default=test_config.test_model_path,
                        help="Path to the model to be tested")

    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of subprocesses to use for data loading")
    parser.add_argument("--use_gpu", type=int, default=1,
                        help="Flag to enable GPU usage if available (1: True, 0: False)")
    parser.add_argument("--is_train", type=int, default=0,
                        help="Flag to indicate if the model is being trained (1: True, 0: False)")

    return parser.parse_args()


if __name__ == '__main__': 
    pass
