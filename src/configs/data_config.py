"""
数据相关设置
"""

# 数据处理常量
S2_min = 0
S2_max = 10000
S2_mean = [0.0943, 0.1160, 0.1117, 0.1502,  0.2550, 0.2617, 0.1975, 0.1478]
S2_std = [0.0561, 0.0565, 0.0614, 0.0602, 0.1067, 0.1078, 0.0790, 0.0733]

MODIS_min = -100
MODIS_max = 16000
MODIS_mean = [0.0595, 0.1577, 0.0420, 0.0608, 0.1135, 0.0707]
MODIS_std = [0.0239, 0.0405, 0.0238, 0.0228, 0.0273, 0.0221]

S1_min = -50
S1_max = 1
S1_mean = [0.7964, 0.6542]
S1_std = [0.0804, 0.0834]
S1_selected_bands = [0, 1]


# 数据日期匹配文件路径
date_match_xlsx_path = "/data/data_date_matching.xlsx"

# 数据根目录
data_root_directory = "/data"

