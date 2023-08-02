import torch

class SensitiveToYouthAndPositiveAgesLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none') # calculate loss per item, not average

    def forward(self, pred, true):
        mse_loss = self.mse_loss(pred, true)
        young_age_penalty = (true < 35).float() * mse_loss
        negative_age_penalty = (pred < 0).float() * mse_loss
        return mse_loss.mean() + young_age_penalty.mean() + negative_age_penalty.mean()
