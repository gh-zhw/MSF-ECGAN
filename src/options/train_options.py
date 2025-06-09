"""
训练命令行参数
"""

import argparse
from configs import train_config


def get_train_opt():
    parser = argparse.ArgumentParser(description="Model Training")

    parser.add_argument("--epochs", type=int, default=train_config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=train_config.BATCH_SIZE,
                        help="Size of each batch")
    parser.add_argument("--lr", type=float, default=train_config.LEARNING_RATE,
                        help="Learning rate for the optimizer")

    parser.add_argument("--recon_mode", type=int, default=1,
                        help="Reconstruction mode "
                             "(1: Double auxiliary data + double reference data, "
                             "2: MODIS auxiliary data + double reference data, "
                             "3: S1 auxiliary data + double reference data, "
                             "4: Only double reference data, "
                             "5: Double auxiliary data + single reference data)")
    parser.add_argument("--gan_mode", type=str, default='lsgan',
                        help="Type of GAN loss (options: 'vanilla', 'lsgan', 'wgangp')")

    parser.add_argument("--lambda_l1", type=int, default=train_config.LAMBDA_L1,
                        help="Weight for L1 loss")
    parser.add_argument("--lambda_sam", type=int, default=train_config.LAMBDA_SAM,
                        help="Weight for SAM loss")

    parser.add_argument("--beta1", type=float, default=train_config.Adam_beta1,
                        help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=train_config.Adam_beta2,
                        help="Beta2 parameter for Adam optimizer")

    parser.add_argument("--with_sam_loss", type=int, default=1,
                        help="Flag to use SAM loss in the options process (1: True, 0: False)")
    parser.add_argument("--sam_unit", type=str, default='rad',
                        help="Unit mode for SAM (options: 'rad', 'deg')")

    parser.add_argument("--valid_freq", type=int, default=1,
                        help="Frequency of validating the model (in number of epochs)")
    parser.add_argument("--print_freq", type=int, default=50,
                        help="Frequency of printing training status and saving logging (in number of iterations)")
    parser.add_argument("--save_freq", type=int, default=50,
                        help="Frequency of saving the model (in number of epochs. 0: only save the best model.)")

    parser.add_argument("--save_dir", type=str, default=train_config.saving_dir,
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default=train_config.logging_dir,
                        help="Directory to save tensorboard logs")

    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of subprocesses to use for data loading")
    parser.add_argument("--use_gpu", type=int, default=1,
                        help="Flag to enable GPU usage if available (1: True, 0: False)")
    parser.add_argument("--is_train", type=int, default=1,
                        help="Flag to indicate if the model is being trained (1: True, 0: False)")

    return parser.parse_args()


if __name__ == '__main__': 
    pass
