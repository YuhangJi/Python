import os
from provider import getPath

# 数据集所在目录
DATASET_PATH = "D:\\DATASET"

# 数据文件指针目录
ALL_FILES_PATH = getPath(DATASET_PATH,
                         "indoor3d_sem_seg_hdf5_data\\all_files.txt")

# 房间号指针目录
ROOM_FILELIST_PATH = getPath(DATASET_PATH,
                             "indoor3d_sem_seg_hdf5_data\\room_filelist.txt")

# batch的数量
BATCH_SIZE = 32

# 每个batch的点数量
NUM_POINT = 4096

# 每个点的属性数量
NUM_ATTRIBUTE = 9

# 点语义数量
NUM_CLASSES = 13

# 属性通道数量
NUM_CHANNEL = 1

# 训练的轮次
EPOCHS = 70

# log目录
LOGS = "logs"

# 检查点目录
SAVE_PATH = os.path.join(LOGS,"pointnet_weights")

# 初始学习率
INITIAL_LEARNING_RATE = 0.0001

# 衰减率
DECAY_RATE = 0.95

# 衰减步数
DECAY_STEPS = 170000

# 验证频率
TEST_FREQUENCE = 1


