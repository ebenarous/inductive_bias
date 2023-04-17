import torch.nn.functional as F
import logging
import torch

def info_nce_loss(out, temperature=0.5, 
                  log=False, logger=None, mode="train"):

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
        logger.info(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        logger.info(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        logger.info(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

    return nll, (sim_argsort == 0).float().mean()
