import argparse
import numpy as np
import pandas as pd
from env import NetworkEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import load_preprocessors


def make_env_from_csv(csv_path, preprocessors_dir=None, episode_length=256):
    df = pd.read_csv(csv_path)
    if preprocessors_dir is not None:
        scaler, encs, le, meta = load_preprocessors(preprocessors_dir)
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
    env = NetworkEnv(X, y, episode_length=episode_length)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--preprocessors', default=None)
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--save', default='saved_models/ppo_detector')
    parser.add_argument('--episode_length', type=int, default=256)
    args = parser.parse_args()

    env = make_env_from_csv(args.data, preprocessors_dir=args.preprocessors, episode_length=args.episode_length)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO('MlpPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f"✅ Saved model to {args.save}")
