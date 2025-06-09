"""
模型训练
"""
import time
from torch.utils.tensorboard import SummaryWriter

from dataset.load_dataset import get_dataloader, create_dataset
from evaluation.metric import calc_metrics
from models.msf_ecgan_model import MSFECGANModel
from models.msf_ecgan_rice import MSFECGAN_rice
from options.train_options import get_train_opt


def train(opt):
    model = MSFECGANModel(opt)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(opt.batch_size,
                                                                         *create_dataset(),
                                                                         opt.num_workers)
    # tensorboard
    writer = SummaryWriter(opt.log_dir)

    best_epoch = 1
    best_val_loss = float('inf')

    cur_iter = 1
    total_iters = opt.epochs * len(train_dataloader)
    for epoch in range(1, opt.epochs + 1):
        epoch_start_time = time.time()

        print("=" * 30 + f" epoch {epoch} " + "=" * 30)
        print("Current G learning rate:", model.optimizer_G.param_groups[0]['lr'])
        print("Current D learning rate:", model.optimizer_D.param_groups[0]['lr'])

        # train
        model.train()
        for train_data_batch in train_dataloader:
            # 优化模型
            loss_dict = model.optimize_model(train_data_batch)

            # 打印 loss & 保存 log
            if cur_iter % opt.print_freq == 0 or cur_iter == total_iters:
                loss_log = " | ".join([f"{key} = {val:.5f}" for key, val in loss_dict.items()])
                print(f"[iteration {cur_iter}/{total_iters}] {loss_log}")

                for key, val in loss_dict.items():
                    writer.add_scalar("train_" + key, val, cur_iter)

            cur_iter += 1

        # validate per valid_freq
        if epoch % opt.valid_freq == 0 or (opt.save_freq > 0 and epoch % opt.save_freq == 0) or epoch == opt.epochs:
            print("Evaluating...")
            model.eval()
            valid_metric_dict = dict()
            for valid_data_batch in valid_dataloader:
                prediction = model.predict(valid_data_batch)
                target = valid_data_batch['S2_target']

                # 计算各项指标
                metric_dict = calc_metrics(prediction, target, denorm=True)
                for key, val in metric_dict.items():
                    if key in valid_metric_dict:
                        valid_metric_dict[key] += val
                    else:
                        valid_metric_dict[key] = val

            for key in valid_metric_dict.keys():
                valid_metric_dict[key] /= len(valid_dataloader)

            print(f"Evaluation metrics: " +
                  " | ".join([f"{k} = {v:.4f}" for k, v in valid_metric_dict.items()]))

            for key, val in valid_metric_dict.items():
                writer.add_scalar("eval_" + key, val, epoch)


            # save model per save_freq
            if opt.save_freq > 0 and epoch % opt.save_freq == 0:
                print("Saving the latest model (epoch %d, cur_iter %d)" % (epoch, cur_iter-1))
                model.save_model(opt.save_dir, epoch, valid_metric_dict)

            # save best model
            if valid_metric_dict['MAE'] < best_val_loss:
                best_epoch = epoch
                best_val_loss = valid_metric_dict['MAE']
                print("Saving the best model (epoch %d, cur_iter %d)" % (epoch, cur_iter-1))
                model.save_model(opt.save_dir, 'best')

        epoch_time_taken = time.time() - epoch_start_time
        estimated_remaining_time = epoch_time_taken * (opt.epochs - epoch)
        print("End of epoch [%d / %d] \t Time Taken: %d sec \t Estimated Time Remaining: %d sec." % (
            epoch, opt.epochs, epoch_time_taken, estimated_remaining_time))

    print("Complete training.")
    print(f"The best model is saved at epoch {best_epoch}.")


if __name__ == '__main__':
    opt = get_train_opt()
    opt.print_freq = 10
    train(opt)
    

