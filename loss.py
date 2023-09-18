import torch.nn.functional as F
import logging
import torch
import cv2
import numpy as np

def info_nce_loss(out, temperature=0.5, 
                  log=False, logger=None, mode="train"):
    # adapted from https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html

    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(out[:, None, :], out[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Get ranking position of positive example
    comb_sim = torch.cat(
        [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
        dim=-1,
    )
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    
    # Logging ranking metrics
    if log:
        logger.info(mode + "_acc_top1: {}".format((sim_argsort == 0).float().mean()))
        logger.info(mode + "_acc_top5: {}".format((sim_argsort < 5).float().mean()))
        logger.info(mode + "_acc_mean_pos: {}".format(1 + sim_argsort.item().float().mean()))

    return nll, (sim_argsort == 0).float().mean()

def compute_saliency_map(outputs, model, input_images, embedding=True, normalized=True):
    """
    Compute the saliency map of an input image with respect to model's prediction.
    
    Args:
    - outputs (torch.Tensor): A torch tensor of shape (BS, X) representing the outputs either before or after the fc layers.
    - model (torch.nn.Module): A trained model.
    - input_images (torch.Tensor): A tensor of shape (BS, C, H, W) representing a batch of images.
    
    Returns:
    - torch.Tensor: A tensor of shape (H, W) representing the saliency map.
    """

    if embedding:
        # Get the L2 Norm of the embedding output
        target_scores = outputs.norm(p=2, dim=1) # TODO: Metric can be modified
    else:
        # Get the max log-probability
        target_scores = outputs.max(dim=1)[0]
    
    # Zero gradients
    model.zero_grad()
    
    # Compute backward pass
    saliency_maps = [] 
    for i in range(input_images.size(0)):
        # Zero out previous gradients
        model.zero_grad()
        input_images.grad = None

        # Backward pass for the specific image
        target_scores[i].backward(retain_graph=True)

        # Compute saliency map for the specific image
        saliency = input_images.grad[i].data.abs().max(dim=0)[0]

        # Convert to probability distribution
        if normalized:
            saliency = saliency / saliency.sum()

        saliency_maps.append(saliency)

    return torch.stack(saliency_maps)

def canny_edge_detector(input_images, low_threshold=75, high_threshold=175):

    edges_batch = []
    for i in range(len(input_images)):
        # Convert tensor to OpenCV image
        np_img = input_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        np_img = (np_img * 255).astype(np.uint8)

        # Convert to grayscale
        gray_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)

        # Reduce noise using Gaussian filter
        blurred_image = cv2.GaussianBlur(gray_img, (7, 7), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

        # Convert back to tensor and append to results
        edges_tensor = torch.from_numpy(edges).float() / 255.0
        edges_batch.append(edges_tensor)
    
    return torch.stack(edges_batch)

def gaussian_kernel(size: int, sigma: float):
    """Generate a 2D Gaussian kernel."""
    coords = torch.arange(size).float() - size // 2
    m, n = torch.meshgrid(coords, coords)
    kernel = torch.exp(-(m ** 2 + n ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel

def edge2blob(tensor_batch, kernel_size=3, sigma=1.0, device='cpu'):
    """Apply Gaussian kernel to pixels with value 1 using convolution for a batch of tensors."""
    # Add one dimension for channel to use F.conv2d
    tensor_batch = tensor_batch[:, None, ...]
    
    kernel = gaussian_kernel(kernel_size, sigma)
    # Add two dimensions for in_channels and out_channels to use F.conv2d
    kernel = kernel[None, None, ...].to(device)
    
    # Apply convolution without padding
    padding = kernel_size // 2
    result = F.conv2d(tensor_batch, kernel, padding=padding)
    
    # Mask the result to only keep values where the original edges were found
    # result = result * tensor_batch

    # Remove the added channel dimension
    result = result[:, 0, ...]
    
    # Normalize
    result = result / result.sum(dim=[1, 2], keepdim=True)
    
    return result

def kl_divergence(P, Q, forward=True):
    """
    Compute the KL divergence between two 2D probability distributions P and Q.
    Forward: D_KL(P || Q) = sum(P * log(P/Q))
    """
    # Ensure the distributions are normalized
    P = P / P.sum()
    Q = Q / Q.sum()

    # Compute KL divergence
    if forward: kl_div = P * (torch.log(P + 1e-10) - torch.log(Q + 1e-10))
    else: kl_div = Q * (torch.log(Q + 1e-10) - torch.log(P + 1e-10))
    
    return kl_div.sum()
