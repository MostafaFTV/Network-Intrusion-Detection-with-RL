"""
Evaluate trained model: compute mean/std of episode rewards and plot ROC curve.
Example:
python evaluate.py --data processed.csv --model saved_models/ppo_detector.zip --preprocessors saved_models/preprocessors
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from stable_baselines3 import PPO
from utils import load_preprocessors


def get_action_probs_from_model(model, obs_batch):
    # obs_batch: numpy array shape (n_samples, n_features)
    # returns probs shape (n_samples, n_actions)
    # Uses internal policy methods to get distribution
    obs_tensor = model.policy.obs_to_tensor(obs_batch)[0]
    dist = model.policy.get_distribution(obs_tensor)
    # اضافه کردن detach() قبل از تبدیل به numpy
    probs = dist.distribution.probs.detach().cpu().numpy()
    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--preprocessors', default=None)
    parser.add_argument('--plot', default='roc.png')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.preprocessors:
        scaler, encs, le, meta = load_preprocessors(args.preprocessors)
        missing_columns = [col for col in meta['feature_names'] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        X = df[meta['feature_names']].values.astype(float)
        X = scaler.transform(X)
        y = df[meta['label_col']].values.astype(int)
    else:
        label_col = df.columns[-1]
        X = df.drop(columns=[label_col]).values.astype(float)
        y = df[label_col].values.astype(int)

    model = PPO.load(args.model)

    # compute predicted probs
    probs = get_action_probs_from_model(model, X.astype(np.float32))
    # predicted label = argmax
    y_pred = probs.argmax(axis=1)

    # compute mean and std of step rewards by simulating episodes of length 1..
    rewards = (y_pred == y).astype(int) * 1 + (y_pred != y).astype(int) * -1
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    print(f"Mean reward: {mean_reward:.4f}, Std reward: {std_reward:.4f}")

    # ROC curve (use probability of class 1)
    if probs.shape[1] >= 2:
        scores = probs[:, 1]
    else:
        # fallback: scores are just predicted labels (coarse)
        scores = y_pred

    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(args.plot)
    print(f"ROC plot saved to {args.plot}")