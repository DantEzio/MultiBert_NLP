import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from random import randint
import numpy as np


class config:
    # 模型构建参数
    def __init__(self, **kwargs):
        # File path
        self.finetuneFile = kwargs['finetuneFile']
        self.pretrainFile = kwargs['pretrainFile']
        self.ResultsFile = kwargs['resultsFile']
        
        # model hyper
        self.initial = kwargs['initial']
        self.model_name = kwargs['model_name']
        self.ref_data = kwargs['ref_data']
        self.max_expand = kwargs['max_expand']

class pretrain_config:
    # 预训练过程使用参数
    def __init__(self, **kwargs):
        # File path
        self.per_device_train_batch_size = kwargs['per_device_train_batch_size']
        self.save_steps = kwargs['save_steps'] #Save model every given steps
        self.num_train_epochs = kwargs['num_train_epochs'] #Number of epochs
        self.use_fp16 = kwargs['use_fp16'] #Set to True, if your GPU supports FP16 operations
        self.max_length = kwargs['max_length'] #Max length for a text input
        self.do_whole_word_mask = kwargs['do_whole_word_mask'] #If set to true, whole words are masked
        self.mlm_prob = kwargs['mlm_prob'] #Probability that a word is replaced by a [MASK] token
