import torch
import numpy as np

class LossAverage:
    """Computes and stores the average and current value for calculating average loss."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)

class DiceAverage:
    """Computes and stores the average Dice coefficient for each class."""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.zeros(self.class_num, dtype=np.float64)
        self.avg = np.zeros(self.class_num, dtype=np.float64)
        self.sum = np.zeros(self.class_num, dtype=np.float64)
        self.count = 0

    def update(self, logits, targets):
        self.value = self.compute_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    def compute_dices(self, logits, targets):
        dices = []
        for class_index in range(targets.size(1)):
            inter = torch.sum(logits[:, class_index] * targets[:, class_index])
            union = torch.sum(logits[:, class_index]) + torch.sum(targets[:, class_index])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.array(dices)

