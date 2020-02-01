import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn as nn

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)

# 终极封装
class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        # gpu 或者 cpu
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        # 是否train
        for net in self.trainer.networks:
            net.train(mode)

# 基类 ddpg 继承该基类
class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    # 外部调用 train 函数
    def train(self, np_batch):
        # 1 训练步数，递增
        self._num_train_steps += 1
        # 2 转换数据为torch格式
        batch = np_to_pytorch_batch(np_batch)
        # 3 训练
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
