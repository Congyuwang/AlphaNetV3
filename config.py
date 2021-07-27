# 数据路径
DATA = "./data/CSI800.zip"

# 时间序列训练参数
HISTORY_LENGTH = 60
TRAIN_LENGTH = 1200
VALIDATE_LENGTH = 150
SAMPLE_STEP = 2

# 神经网络训练参数
EPOCHS = 150
BATCH_SIZE = 500
DROPOUT = 0.0
L2 = 0.001
LEARNING_RATE = 0.0001

# early-stopping
EARLY_STOPPING_PATIENCE = 100
# 注意：
# 当early stopping是False, EARLY_STOPPING_PATIENCE

# 每个rolling period开始的日期
ROLLING_BEGINNING_LIST = [20110131,
                          20110731,
                          20120131,
                          20120731,
                          20130131,
                          20130731,
                          20140131,
                          20140731,
                          20150131,
                          20150731,
                          20160131,
                          20160731,
                          20170131,
                          20170731,
                          20180131,
                          20180731]

# 多次training，降低初始化偶然性的影响
TRAINING_ID = [0, 1, 2, 3, 4]
