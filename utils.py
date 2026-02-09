import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def save_preprocessors(out_dir, scaler, encoders, label_encoder=None, meta=None):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(out_dir, "encoders.joblib"))
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(out_dir, "label_encoder.joblib"))
    if meta is not None:
        joblib.dump(meta, os.path.join(out_dir, "meta.joblib"))


def load_preprocessors(out_dir):
    required_files = ["scaler.joblib", "encoders.joblib"]
    for file in required_files:
        if not os.path.exists(os.path.join(out_dir, file)):
            raise FileNotFoundError(f"Required file '{file}' not found in directory '{out_dir}'")

    scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))
    encoders = joblib.load(os.path.join(out_dir, "encoders.joblib"))
    label_encoder = None
    meta = None
    le_path = os.path.join(out_dir, "label_encoder.joblib")
    meta_path = os.path.join(out_dir, "meta.joblib")
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
    return scaler, encoders, label_encoder, meta


def detect_label_column(df, given_label=None):
    if given_label is not None and given_label in df.columns:
        return given_label
    for cand in ["label", "Label", "target", "class"]:
        if cand in df.columns:
            return cand

    return df.columns[-1]
