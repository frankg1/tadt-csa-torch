import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

DATASETS = {
    'kuairand': 'dt_format_datasets/kuairand_trajectories.json',
    'ml': 'dt_format_datasets/ml_trajectories.json',
    'retailrocket': 'dt_format_datasets/retailrocket_trajectories.json',
    'netflix': 'dt_format_datasets/netflix_trajectories.json'
}

def compute_rtg_and_advantage_trend_from_rewards(reward_seq, gamma=0.9):
    """
    reward_seq: [L] or [B, L]
    返回: rtg, adv_trend, shape同reward_seq
    """
    if isinstance(reward_seq, list):
        reward_seq = torch.tensor(reward_seq, dtype=torch.float)
    if reward_seq.ndim == 1:
        reward_seq = reward_seq.unsqueeze(0)
    B, L = reward_seq.shape
    device = reward_seq.device
    rtg = torch.zeros_like(reward_seq)
    for t in reversed(range(L)):
        if t == L - 1:
            rtg[:, t] = reward_seq[:, t]
        else:
            rtg[:, t] = reward_seq[:, t] + gamma * rtg[:, t + 1]
    adv_trend = torch.zeros_like(reward_seq)
    for t in range(1, L):
        diffs = rtg[:, 1:t+1] - rtg[:, :t]
        gammas = gamma ** torch.arange(t - 1, -1, -1).float().to(device)
        weighted_diffs = diffs * gammas.view(1, -1)
        adv_trend[:, t] = weighted_diffs.sum(dim=1)
    return rtg.squeeze(0).cpu().numpy(), adv_trend.squeeze(0).cpu().numpy()

def plot_and_save_hist(data, title, save_path, bins=100):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_dataset(dataset_name, json_path):
    print(f"\n==== Analyzing {dataset_name} ====")
    with open(json_path, 'r', encoding='utf-8') as f:
        trajectories = json.load(f)
    rewards, rtgs, advs = [], [], []
    for user, traj in trajectories.items():
        reward_seq = traj['reward'][:traj['length']]
        rewards.extend(reward_seq)
        rtg_seq, adv_seq = compute_rtg_and_advantage_trend_from_rewards(reward_seq)
        rtgs.extend(rtg_seq.tolist())
        advs.extend(adv_seq.tolist())
    rewards = np.array(rewards)
    rtgs = np.array(rtgs)
    advs = np.array(advs)
    print(f"Reward: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}, std={rewards.std():.4f}, 25%={np.percentile(rewards,25):.4f}, 50%={np.percentile(rewards,50):.4f}, 75%={np.percentile(rewards,75):.4f}")
    print(f"RTG:    min={rtgs.min():.4f}, max={rtgs.max():.4f}, mean={rtgs.mean():.4f}, std={rtgs.std():.4f}, 25%={np.percentile(rtgs,25):.4f}, 50%={np.percentile(rtgs,50):.4f}, 75%={np.percentile(rtgs,75):.4f}")
    print(f"Adv:    min={advs.min():.4f}, max={advs.max():.4f}, mean={advs.mean():.4f}, std={advs.std():.4f}, 25%={np.percentile(advs,25):.4f}, 50%={np.percentile(advs,50):.4f}, 75%={np.percentile(advs,75):.4f}")
    os.makedirs('rtg_adv_analysis', exist_ok=True)
    plot_and_save_hist(rewards, f"{dataset_name} Reward Distribution", f"rtg_adv_analysis/{dataset_name}_reward_hist.png")
    plot_and_save_hist(rtgs, f"{dataset_name} RTG Distribution", f"rtg_adv_analysis/{dataset_name}_rtg_hist.png")
    plot_and_save_hist(advs, f"{dataset_name} Advantage Distribution", f"rtg_adv_analysis/{dataset_name}_adv_hist.png")

if __name__ == "__main__":
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        analyze_dataset(name, path)
    print("\nAll done! Plots saved in rtg_adv_analysis/")