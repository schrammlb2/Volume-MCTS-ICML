import torch

def DiagonalNormal_kl_divergence(dist_1, dist_2):
	mu_1, sigma_1 = dist_1
	mu_2, sigma_2 = dist_2

	ret_val = 1/2 * torch.log(sigma_2/sigma_1) + (sigma_1**2 + (mu_1 - mu_2)**2)/sigma_2**2 - 1/2
	if ret_val.dim() == 1: 
		return ret_val
	elif ret_val.dim() == 2:
		return torch.sum(ret_val, dim=-1)
	else: 
		import ipdb
		ipdb.set_trace()

def DiagonalNormal_rkl_divergence(dist_1, dist_2):
	return DiagonalNormal_kl_divergence(dist_2, dist_1)

def DiagonalNormal_square_divergence(dist_1, dist_2):
	mu_1, log_sigma_1 = dist_1
	mu_2, log_sigma_2 = dist_2
	mu_diff = (mu_1 - mu_2)**2/((torch.exp(log_sigma_1)**2 + torch.exp(log_sigma_2)**2).detach())
	sigma_diff= (log_sigma_1 - log_sigma_2)**2
	ret_val = mu_diff + sigma_diff
	if ret_val.dim() == 1: 
		return ret_val
	elif ret_val.dim() == 2:
		return torch.sum(ret_val, dim=-1)
	else: 
		import ipdb
		ipdb.set_trace()

def DiagonalNormal_js_divergence_log_std(dist_1, dist_2):
	mu_1, log_sigma_1 = dist_1
	mu_2, log_sigma_2 = dist_2
	sigma_1 = torch.exp(log_sigma_1)
	sigma_2 = torch.exp(log_sigma_2)
	clamped_sigma_1 = torch.exp(torch.clamp(log_sigma_1, min=-5, max=2))
	clamped_sigma_2 = torch.exp(torch.clamp(log_sigma_2, min=-5, max=2))

	ret_val_forward  = 1/2 * (log_sigma_2 - log_sigma_1) + (sigma_1**2 + (mu_1 - mu_2)**2)/clamped_sigma_2**2 - 1/2
	ret_val_backward = 1/2 * (log_sigma_1 - log_sigma_2) + (sigma_2**2 + (mu_2 - mu_1)**2)/clamped_sigma_1**2 - 1/2
	ret_val = ret_val_forward + ret_val_backward
	if ret_val.dim() == 1: 
		return ret_val
	elif ret_val.dim() == 2:
		return torch.sum(ret_val, dim=-1)
	else: 
		import ipdb
		ipdb.set_trace()
