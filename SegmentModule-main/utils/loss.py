import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        dice = self._compute_dice(pred, target)
        return torch.clamp((1 - dice).mean(), 0, 1)

    def _compute_dice(self, pred, target):
        dice = 0.
        for i in range(pred.size(1)):
            intersection = (pred[:, i] * target[:, i]).sum(dim=(1, 2, 3))
            union = pred[:, i].pow(2).sum(dim=(1, 2, 3)) + target[:, i].pow(2).sum(dim=(1, 2, 3))
            dice += 2 * intersection / (union + self.smooth)
        return dice / pred.size(1)

class ELDiceLoss(DiceLoss):
    def forward(self, pred, target):
        dice = self._compute_dice(pred, target)
        return torch.clamp(torch.pow(-torch.log(dice + 1e-5), 0.3).mean(), 0, 2)

class HybridLoss(DiceLoss):
    def __init__(self, bce_weight=1.0, smooth=1):
        super().__init__(smooth)
        self.bce_loss = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        dice = self._compute_dice(pred, target)
        dice_loss = torch.clamp((1 - dice).mean(), 0, 1)
        return dice_loss + self.bce_loss(pred, target) * self.bce_weight

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        jaccard = self._compute_jaccard(pred, target)
        return torch.clamp((1 - jaccard).mean(), 0, 1)

    def _compute_jaccard(self, pred, target):
        jaccard = 0.
        for i in range(pred.size(1)):
            intersection = (pred[:, i] * target[:, i]).sum(dim=(1, 2, 3))
            union = pred[:, i].pow(2).sum(dim=(1, 2, 3)) + target[:, i].pow(2).sum(dim=(1, 2, 3)) - intersection
            jaccard += intersection / (union + self.smooth)
        return jaccard / pred.size(1)

class SSLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        loss = self._compute_ss(pred, target)
        return loss / pred.size(1)

    def _compute_ss(self, pred, target):
        loss = 0.
        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - target[:, i]).pow(2) * target[:, i]).sum(dim=(1, 2, 3)) / (self.smooth + target[:, i].sum(dim=(1, 2, 3)))
            s2 = ((pred[:, i] - target[:, i]).pow(2) * (1 - target[:, i])).sum(dim=(1, 2, 3)) / (self.smooth + (1 - target[:, i]).sum(dim=(1, 2, 3)))
            loss += 0.05 * s1 + 0.95 * s2
        return loss

class TverskyLoss(nn.Module):
    def __init__(self, smooth=1, alpha=0.3, beta=0.7):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        tversky = self._compute_tversky(pred, target)
        return torch.clamp((1 - tversky).mean(), 0, 2)

    def _compute_tversky(self, pred, target):
        tversky = 0.
        for i in range(pred.size(1)):
            tp = (pred[:, i] * target[:, i]).sum(dim=(1, 2, 3))
            fp = (pred[:, i] * (1 - target[:, i])).sum(dim=(1, 2, 3))
            fn = ((1 - pred[:, i]) * target[:, i]).sum(dim=(1, 2, 3))
            tversky += tp / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return tversky / pred.size(1)
