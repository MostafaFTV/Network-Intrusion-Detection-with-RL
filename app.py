import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.exceptions import UndefinedMetricWarning
from stable_baselines3 import PPO
from utils import load_preprocessors  
import warnings

@st.cache_resource
def load_model_and_preprocessors(model_path, preprocessors_path):
    model = PPO.load(model_path)
    scaler, encs, le, meta = load_preprocessors(preprocessors_path)
    return model, scaler, encs, le, meta

def safe_label_transform(le, labels, unknown_label="normal"):
    if le is None:
        raise ValueError("LabelEncoder is not initialized. Please check the preprocessors.")
    known_classes = set(le.classes_)
    safe_labels = [
        label if label in known_classes else unknown_label
        for label in labels
    ]
    return le.transform(safe_labels)

def main():
    st.title("Network Intrusion Detection - PPO Model")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Raw data preview:")
        st.dataframe(df.head())

        model, scaler, encs, le, meta = load_model_and_preprocessors(
            'saved_models/ppo_detector.zip',
            'saved_models/preprocessors'
        )

        for col, encoder in encs.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        X = df[meta['feature_names']].values.astype(float)
        X = scaler.transform(X)

        if meta['label_col'] in df.columns:
            y_true = df[meta['label_col']].astype(str).apply(lambda x: 0 if x.lower() == "normal" else 1).values
        else:
            st.error(f"Label column '{meta['label_col']}' not found in uploaded data.")
            return

        obs = X.astype(np.float32)
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()
        y_pred = probs.argmax(axis=1)

        st.write("Prediction Results:")
        results_df = pd.DataFrame({
            'True Label': y_true,
            'Predicted Label': y_pred
        })
        st.dataframe(results_df)

        rewards = (y_pred == y_true).astype(int) * 1 + (y_pred != y_true).astype(int) * -1
        st.write(f"Mean reward: {rewards.mean():.4f}")
        st.write(f"Std reward: {rewards.std():.4f}")

        st.write("### Model Evaluation Results")
        st.write(f"**Mean Reward:** {rewards.mean():.4f}")
        st.write(f"**Std Reward:** {rewards.std():.4f}")

        st.write("### True Labels and Predictions")
        st.write(f"**Classes in LabelEncoder:** {list(le.classes_)}")
        st.write(f"**Unique Classes in y_true:** {list(map(int, set(y_true)))}")

        unique, counts = np.unique(y_true, return_counts=True)
        distribution_y_true = {int(k): int(v) for k, v in zip(unique, counts)}
        st.write("**Distribution in y_true:**")
        st.json(distribution_y_true)

        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        distribution_y_pred = {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}
        st.write("**Distribution in y_pred:**")
        st.json(distribution_y_pred)

        if probs.shape[1] >= 2:
            scores = probs[:, 1]
        else:
            scores = y_pred
        if len(set(y_true)) < 2:
            st.warning("AUC cannot be calculated because y_true contains only one class.")
        else:
            st.write(f"y_true: {y_true}")
            st.write(f"y_pred: {y_pred}")

            if le is not None:
                st.write(f"Classes in LabelEncoder: {le.classes_}")
            else:
                st.error("LabelEncoder is not initialized.")

            st.write(f"Unique classes in y_true: {set(y_true)}")

            unique, counts = np.unique(y_true, return_counts=True)
            st.write(f"Distribution in y_true: {dict(zip(unique, counts))}")

            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            st.write(f"Distribution in y_pred: {dict(zip(unique_pred, counts_pred))}")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], linestyle='--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC curve')
                ax.legend(loc='lower right')
                st.pyplot(fig)

if __name__ == "__main__":
    main()

