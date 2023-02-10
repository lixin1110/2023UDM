import torch
from . import soft_dtw
from . import path_soft_dtw 
from .soft_dtw_cuda import SoftDTW

def dilate_loss(outputs, targets, theta, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	#loss_shape = softdtw_batch(D,gamma)
	sdtw = SoftDTW(use_cuda=True, gamma=gamma)
	loss_shape = sdtw(outputs,targets)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = theta*loss_shape+ (1-theta)*loss_temporal
	return loss, loss_shape, loss_temporal