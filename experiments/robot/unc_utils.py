from typing import Any
import numpy as np
import scipy
import torch
import torch.nn.functional as F

from sklearn.cluster import AgglomerativeClustering


def logits2entropy(logits, dim=-1):
    p = F.softmax(logits, dim=dim)
    logp = F.log_softmax(logits, dim=dim)
    return - torch.sum(p * logp, dim=dim)


def compute_token_uncertainty_metrics(
    generated_outputs: dict,
    model: Any,
):
    metrics = {}
    if "logits" in generated_outputs:
        logits = generated_outputs["logits"]
        if type(logits) is list or type(logits) is tuple:
            logits = torch.cat(logits, dim=0)
        vocab_size = model.config.text_config.vocab_size - model.config.pad_to_multiple_of
        n_action_bins = model.config.n_action_bins
        
        # vocab_size - 1 is the rightmost bin. 
        logits = logits[:, vocab_size - n_action_bins + 1 : vocab_size + 1] # (n_actions, n_bins)
        token_prob = F.softmax(logits, dim=-1) # (n_actions, n_bins)
        selected_token_prob = token_prob.max(dim=-1).values # (n_actions)
        entr = logits2entropy(logits) # (c_actions)
        
        metrics["mean_token_entropy"] = entr.mean().item()
        metrics["max_token_entropy"] = entr.max().item()
        metrics["mean_token_prob"] = selected_token_prob.mean().item()
        metrics["max_token_prob"] = selected_token_prob.max().item()
        
        # Also log the per-token entropy
        for i in range(entr.shape[0]):
            metrics[f"token_{i}_entropy"] = entr[i].item()
            metrics[f"token_{i}_prob"] = selected_token_prob[i].item()

    return metrics


def compute_samples_uncertainty_metrics(
    actions: np.ndarray,
):
    '''
    actions in shape of (n_samples, action_dim)
    '''
    action_cov = np.cov(actions, rowvar=False)
    total_var = np.trace(action_cov)
    general_var = np.linalg.det(action_cov)
    pos_var = np.trace(action_cov[:3, :3])
    rot_var = np.trace(action_cov[3:6, 3:6])
    gripper_var = np.trace(action_cov[6:, 6:])
    
    metrics = {
        "total_var": total_var,
        "general_var": general_var,
        "pos_var": pos_var,
        "rot_var": rot_var,
        "gripper_var": gripper_var,
    }
    
    # Perform linkage clustering for computing entropy metrics
    linkage = AgglomerativeClustering(n_clusters=None, distance_threshold=0.01)
    cluster_labels = linkage.fit_predict(actions)
    _, counts = np.unique(cluster_labels, return_counts=True)
    entropy = scipy.stats.entropy(counts)
    metrics['entropy_linkage.01'] = entropy
    
    linkage = AgglomerativeClustering(n_clusters=None, distance_threshold=0.05)
    cluster_labels = linkage.fit_predict(actions)
    _, counts = np.unique(cluster_labels, return_counts=True)
    entropy = scipy.stats.entropy(counts)
    metrics['entropy_linkage.05'] = entropy
    
    return metrics, cluster_labels