import os
import pandas as pd
import torch
import random
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter

class LoggerBase:
    def __init__(self, save_path, save_name):
        self.log = None
        self.summary = None
        self.save_path = save_path
        self.save_name = save_name

    def update_csv(self, item):
        tmp = pd.DataFrame(item, index=[0])
        if self.log is not None:
            self.log = pd.concat([self.log, tmp], ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv(os.path.join(self.save_path, f'{self.save_name}.csv'), index=False)

    def update_tensorboard(self, item):
        if self.summary is None:
            self.summary = SummaryWriter(self.save_path)
        epoch = item.get('epoch', 0)
        for key, value in item.items():
            if key != 'epoch':
                self.summary.add_scalar(key, value, epoch)

class TrainLogger(LoggerBase):
    def update(self, epoch, train_log, val_log):
        item = OrderedDict({'epoch': epoch})
        item.update(train_log)
        item.update(val_log)
        print("\033[0;33mTrain:\033[0m", train_log)
        print("\033[0;33mValid:\033[0m", val_log)
        self.update_csv(item)
        self.update_tensorboard(item)

class TestLogger(LoggerBase):
    def update(self, name, log):
        item = OrderedDict({'img_name': name})
        item.update(log)
        print("\033[0;33mTest:\033[0m", log)
        self.update_csv(item)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def dict_round(dic, num):
    return {key: round(value, num) for key, value in dic.items()}
