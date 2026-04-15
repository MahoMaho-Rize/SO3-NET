import torch
import torch.nn as nn

# binary cross-entropy loss
def bce_loss(pred, gold):
    bce_fn = nn.BCELoss(reduce=True, size_average=True)
    bce_loss = bce_fn(pred, gold.float())
    return bce_loss

# fitting residual loss
def fr_loss(pred, points, coef_d):

    pred_select = pred > 0.5 #bx2048 [[0,1,0,0,1,1,...],[...],...]
    min_y = coef_d.squeeze(dim=1) #b
    residual = abs(points[:,:,1] - min_y.unsqueeze(1)) #bx2048

    fitting_residual = residual * pred * pred_select #bx2048

    smoothl1_fn = nn.SmoothL1Loss(reduction='none')

    fr_loss = smoothl1_fn(fitting_residual, torch.zeros_like(fitting_residual)) #bx2048
    fr_loss = torch.mean((torch.sum(fr_loss, dim=1) + (torch.sum(pred_select, dim=1) == 0) * 2.0) \
            / torch.max(torch.sum(pred, dim=1), torch.tensor([1]).cuda()))

    return fr_loss

 
