"""
模型测试
"""
import time

from dataset.load_dataset import get_dataloader, create_dataset
from evaluation.metric import calc_metrics
from models.msf_ecgan_model import MSFECGANModel
from options.test_options import get_test_opt

if __name__ == '__main__':
    opt = get_test_opt()

    model = MSFECGANModel(opt)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(opt.batch_size,
                                                                         *create_dataset(),
                                                                         opt.num_workers)

    print("Testing...")
    model.eval()
    start_time = time.time()
    test_metric_dict = dict()
    for test_data_batch in test_dataloader:
        prediction = model.predict(test_data_batch)
        target = test_data_batch['S2_target']

        # 计算各项指标
        metric_dict = calc_metrics(prediction, target, denorm=True)
        for key, val in metric_dict.items():
            if key in test_metric_dict:
                test_metric_dict[key] += val
            else:
                test_metric_dict[key] = val

    for key in test_metric_dict.keys():
        test_metric_dict[key] /= len(test_dataloader)

    print(f"Test metrics: " +
          " | ".join([f"{k} = {v:.4f}" for k, v in test_metric_dict.items()]))
    print("End of testing \t Time Taken: %d sec." % (time.time() - start_time))
