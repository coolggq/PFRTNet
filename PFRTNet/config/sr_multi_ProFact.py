from yacs.config import CfgNode as CN

# config definition
_C = CN()

_C.INPUT_SIZE = 160
_C.SCALE = 2

_C.SEED = 42

_C.dist_url = 'env://'
_C.world_size = 1

# dataset config
_C.DATASET = CN()
_C.DATASET.ROOT = '/home/gegq/data/BraTS/h5'  # the root of dataset
_C.DATASET.CHALLENGE = 'singlecoil'  # the task of ours, singlecoil or multicoil
_C.DATASET.MODE = 'train'  # train or test

_C.TRANSFORMS = CN()
_C.TRANSFORMS.MASKTYPE = 'equispaced'  # "random" or "equispaced"
_C.TRANSFORMS.CENTER_FRACTIONS = [0.08]
_C.TRANSFORMS.ACCELERATIONS = [8]

# model config
_C.MODEL = CN()
_C.MODEL.INPUT_DIM = 1   # the channel of input,  head部分
_C.MODEL.OUTPUT_DIM = 1   # the channel of output，
_C.MODEL.HEAD_HIDDEN_DIM = 16  # the hidden dim of Head
_C.MODEL.TRANSFORMER_DEPTH = 4  # the depth of the transformer
_C.MODEL.TRANSFORMER_NUM_HEADS = 4 # the head's num of multi head attention
_C.MODEL.TRANSFORMER_MLP_RATIO = 3  # the MLP RATIO Of transformer
_C.MODEL.TRANSFORMER_EMBED_DIM = 256  # the EMBED DIM of transformer
_C.MODEL.P1 = 8
_C.MODEL.P2 = 16
_C.MODEL.ver = 'b0'

_C.MODEL.CTDEPTH = 4

_C.MULTI = CN()

# the solver config

_C.SOLVER = CN()
_C.SOLVER.DEVICE = 'cuda'
_C.SOLVER.DEVICE_IDS = [0]  # if [] use cpu, else gpu 0, 1
_C.SOLVER.LR = 4e-4 # 5e-5
_C.SOLVER.WEIGHT_DECAY = 4e-4
_C.SOLVER.LR_DROP = []
_C.SOLVER.BATCH_SIZE = 8 # 4
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.PRINT_FREQ = 10 # args.SOLVER.PRINT_FREQ 是一个通常用于控制在训练过程中打印进度信息的参数。具体来说，它决定了每隔多少个训练步（或轮次）打印一次训练状态或进度信息。

# the others config
_C.RESUME = '/home/gegq/FORsr/weights_change_CMF6/checkpoint0050.pth'  # model resume path
_C.OUTPUTDIR = './weights_change_CMF7'  # the model output dir
_C.TEST_OUTPUTDIR = 'outputs/change_CMF'

#the train configs
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 100 # the train epochs

_C.WORK_TYPE = 'sr' # 'sr'
_C.USE_CL1_LOSS = False     #
_C.USE_MULTI_MODEL = False

