import torch
from loss.dilate import dilate_loss
from loss.pinball import pinball_loss

class Pin_DTW():
    def __init__(self, alpha=0.5, tau=0.6, theta=0.8, gamma=1.0):
        super(Pin_DTW, self).__init__()
        self.pin = pinball_loss
        self.dtw = dilate_loss
        self.alpha = alpha
        self.tau = tau
        self.theta = theta
        self.gamma = gamma

    def forward(self, pred, actual, device):
        pinball_loss = self.pin(pred, actual, self.tau)
        dil_loss, _, _ = self.dtw(pred, actual, theta=self.theta, gamma=self.gamma, device=device)

        # 计算总loss
        loss = torch.mean(self.alpha*pinball_loss + (1-self.alpha)*dil_loss)
        loss.requires_grad_(True)
        return loss