import torch
from torch import nn
from utils.soft_skel import soft_skel
from utils.misc import get_loss

'''
Loss functions for segmentation tasks.
'''

def dice(input, target, epsilon=1e-6):
    intersection = (input * target).sum(1)
    denominator = (input * input).sum(1) + (target * target).sum(1)
    dice = 2 * (intersection / denominator.clamp(min=epsilon))
    return dice


def cldice(input, input_skel, target, target_skel, epsilon=1e-6):
    tprec = ((input_skel * target).sum(1) + epsilon) / (input_skel.sum(1) + epsilon)
    tsens = ((target_skel * input).sum(1) + epsilon) / (target_skel.sum(1) + epsilon)
    cl_dice = 2 * tprec * tsens / (tprec + tsens)
    return cl_dice


class DiceLoss(nn.Module):
    '''
    Dice Loss as formulated by Milletari et al. (2016)

    https://arxiv.org/abs/1606.04797
    '''
    def __init__(self, epsilon=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                .format(target.size(), input.size()))
        input = torch.sigmoid(input)
        batch_size = input.size(0)
        input = input.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        loss = 1. - dice(input, target, self.epsilon)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalTverskyLoss(nn.Module):
    '''
    Focal Tversky Loss proposed by Abraham and Khan (2018)

    https://arxiv.org/pdf/1810.07842.pdf

    Compared to Dice loss, this allows for better tuning of precision and recall
    and emphasizes errors on hard examples. It has been shown to work well for
    medical image segmentation tasks with small ROIs so could be worth testing
    as an alternative loss for U-Net/CasNet architectures.
    '''
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1, reduction='mean'):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, input, target):

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                .format(target.size(), input.size()))
        input = torch.sigmoid(input)

        batch_size = input.size(0)
        input = input.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)

        # True Positives, False Positives & False Negatives
        tp = torch.sum(input * target, 1)
        fp = torch.sum(input * (1-target), 1)
        fn = torch.sum((1-input) * target, 1)

        # Compute index and focal loss
        tversky = (tp + self.smooth) / (tp + self.alpha*fn + self.beta*fp + self.smooth)
        loss = torch.pow(1 - tversky, self.gamma)

        # Reduce loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class CLDice(nn.Module):
    '''
    Soft centerline-Dice loss function proposed by Shit et al. (2020)

    https://arxiv.org/abs/2003.07311
    '''

    def __init__(self, alpha=0.5, iter=3, epsilon=1e-6, reduction='mean'):
        super(CLDice, self).__init__()
        self.alpha = 0.5
        self.iter = iter
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, input, target):

        input = torch.sigmoid(input)
        input_skel = soft_skel(input, self.iter)
        target_skel = soft_skel(target, self.iter)

        batch_size = input.size(0)

        input = input.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        input_skel = input_skel.reshape(batch_size, -1)
        target_skel = target_skel.reshape(batch_size, -1)

        dice_loss = 1. - dice(input, target, self.epsilon)
        cldice_loss = 1. - cldice(input, input_skel, target, target_skel, self.epsilon)
        loss = self.alpha*cldice_loss + (1. - self.alpha)*dice_loss

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class CascadedCLDice(nn.Module):
    '''
    CasNet provides 2 outputs: a predicted segmentation and a predicted centerline.
    This loss function works like clDice, but rather than using soft-skeletonization
    to yield the predicted and ground truth centerlines, it uses the centerline detection
    network's prediction and target.
    '''

    def __init__(self, alpha=0.5, epsilon=1e-6, reverse=False, reduction='mean'):
        super(CascadedCLDice, self).__init__()
        self.reverse = reverse
        self.alpha = alpha
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs, targets):
        assert len(inputs) == len(targets) == 2
        if self.reverse:
            input = torch.sigmoid(inputs[1])
            input_skel = torch.sigmoid(inputs[0])
        else:
            input = torch.sigmoid(inputs[0])
            input_skel = torch.sigmoid(inputs[1])
        target = targets[0]
        target_skel = targets[1]

        batch_size = input.size(0)

        input = input.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)
        input_skel = input_skel.reshape(batch_size, -1)
        target_skel = target_skel.reshape(batch_size, -1)

        cldice_loss = 1. - cldice(input, input_skel, target, target_skel, self.epsilon)
        self_dice_loss = 1. - dice(input, input_skel, self.epsilon)
        loss = self.alpha*cldice_loss + (1. - self.alpha)*self_dice_loss

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class MultiInputLoss(nn.Module):
    '''
    Given a list of inputs and list of targets, apply loss
    function to each (input, target) pair and return average.

    TODO: Add option to weight each term differently, as well as
    report separately so that their relative contributions to overall
    loss can be tracked.
    '''
    def __init__(self, loss_fn, **kwargs):
        super(MultiInputLoss, self).__init__()
        self.loss_fn = get_loss(loss_fn, **kwargs)

    def forward(self, inputs, targets):
        assert len(inputs) == len(targets)
        loss = 0.
        for i, _input in enumerate(inputs):
            try:
                loss += self.loss_fn(_input, targets[i])
            except:
                print(_input, targets[i])
                raise
        return loss/(i+1)
