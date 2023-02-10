import torch

def pinball_loss(forec, actual, Tau):
        Tau = 0.6
        C = forec>actual
        M1 = (1-Tau) * (forec-actual)
        M2 = Tau * (actual-forec)
        M_loss = torch.where(C, M1, M2)
        pin_loss = torch.mean(M_loss, dim=1).squeeze(1)

        return pin_loss