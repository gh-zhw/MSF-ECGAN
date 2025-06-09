"""
训练参数设置
"""

EPOCHS = 300
BATCH_SIZE = 4
LEARNING_RATE = 2e-4

# 损失系数
LAMBDA_L1 = 100
LAMBDA_SAM = 10

# Adam 优化器参数
Adam_beta1 = 0.5
Adam_beta2 = 0.999

# 模型保存文件夹路径
saving_dir = "/result/well-trained-models/"

# 日志文件夹路径
logging_dir = "logs"
