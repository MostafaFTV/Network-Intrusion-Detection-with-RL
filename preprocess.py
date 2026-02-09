"""
Usage examples:
python preprocess.py --input data.csv --output processed.csv --outdir saved_models/preprocessors

The script will:
- auto-detect label column (or take --label_col)
- encode textual features with LabelEncoder
- convert target to binary (if 'normal' appears it will map normal->0 else use label encoder)
- normalize features with MinMaxScaler
- save scaler + encoders to outdir
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os
from utils import save_preprocessors, detect_label_column


def preprocess_dataframe(df, label_col=None):
    label_col = detect_label_column(df, label_col)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in the dataset.")
    print(f"Using label column: {label_col}")
    y = df[label_col].copy()
    X = df.drop(columns=[label_col]).copy()

    encoders = {}
    for col in X.select_dtypes(include=[object, 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"Encoded column {col} -> classes={list(le.classes_)[:6]}{'...' if len(le.classes_)>6 else ''}")

    label_encoder = None
    if y.dtype == object or y.dtype.name == 'category':
        lowers = y.astype(str).str.lower()
        unique = set(lowers.unique())
        if 'normal' in unique:
            print("Detected 'normal' class in labels — mapping 'normal'->0 and others->1")
            y_bin = (lowers != 'normal').astype(int)
            label_encoder = LabelEncoder()
            label_encoder.fit(['normal', 'attack'])  
        else:
            print("Label column is non-numeric and does not contain 'normal' — applying LabelEncoder")
            label_encoder = LabelEncoder()

            all_classes = np.unique(y.astype(str))
            label_encoder.fit(all_classes)

            y_bin = label_encoder.transform(y.astype(str))
    else:
        if set(np.unique(y)) <= {0, 1}:
            y_bin = y.astype(int)
        else:
            print("Warning: numeric labels with more than 2 values — leaving as-is.")
            y_bin = y


    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values)

    meta = {
        'feature_names': list(X.columns),
        'label_col': label_col,
    }
    return X_scaled.astype(np.float32), y_bin.astype(int), scaler, encoders, label_encoder, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--label_col', type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    X_scaled, y_bin, scaler, encoders, label_encoder, meta = preprocess_dataframe(
        df, label_col=args.label_col
    )

    save_preprocessors(
        out_dir=args.outdir,
        scaler=scaler,
        encoders=encoders,
        label_encoder=label_encoder,
        meta=meta
    )

    if args.output:
        out_df = pd.DataFrame(X_scaled, columns=meta['feature_names'])
        out_df[meta['label_col']] = y_bin
        out_df.to_csv(args.output, index=False)
        print(f"Wrote processed CSV to {args.output}")

    print("Preprocessing done. Preprocessors saved to:", args.outdir)


