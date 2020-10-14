import os
from provider import getPath

# 数据集所在目录
DATASET_PATH = "D:\\jiyuhang\\CH-ArchitectreSegmentation\\PointNet\\dataset\\original"

# 数据文件指针目录
ALL_FILES_PATH = getPath(DATASET_PATH,
                         "all_file_names.txt")

# 房间号指针目录
ROOM_FILELIST_PATH = getPath(DATASET_PATH,
                             "all_batch_names.txt")
# 原始点云分块后未做标准化的坐标 主要是用于可视化
ORIGINAL_COORDINATES_PATH = "data_blocked_coordinates\\original"

# batch的数量
BATCH_SIZE = 60

# 每个batch的点数量
NUM_POINT = 4096

# 每个点的属性数量
NUM_ATTRIBUTE = 6

# 点语义数量
NUM_CLASSES = 7

# 属性通道数量
NUM_CHANNEL = 1

# 训练的轮次
EPOCHS = 25

# log目录
LOGS = "logs"

# 检查点目录
SAVE_PATH = os.path.join(LOGS,"pointnet_weights")

# LOG文件
LOGS_DIR = getPath
# 初始学习率
INITIAL_LEARNING_RATE = 0.0001

# 衰减率
DECAY_RATE = 0.5

# 衰减步数
DECAY_STEPS = 150000

# 验证频率
TEST_FREQUENCE = 1


