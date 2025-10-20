# Network-Intrusion-Detection-with-RL
Reinforcement Learning-based Network Intrusion Detection System using PPO algorithm and KDD99 dataset. Detects and adapts to cyberattacks in real-time.
# 🛡️ Network Intrusion Detection System (NIDS) using Reinforcement Learning (PPO)

A Reinforcement Learning-based Network Intrusion Detection System (NIDS) that detects and classifies network attacks such as **DDoS**, **Port Scanning**, and **Brute Force** using the **KDD99 dataset**.  
The system learns adaptive defense strategies through **Proximal Policy Optimization (PPO)** — a powerful RL algorithm.

---

## 🚀 Overview
Traditional intrusion detection systems rely on static rules and signatures.  
This project introduces a **self-learning NIDS** capable of:
- Detecting multiple attack types in real-time.
- Learning optimal defense policies through interaction.
- Adapting to evolving threats over time.

---

## 🧠 Features
- 🔍 **Attack detection**: DDoS, Port Scanning, Brute Force, Insider Threats  
- ⚙️ **Reinforcement Learning (PPO)**: adaptive and self-improving defense agent  
- 📊 **Data Preprocessing**: ARFF → CSV conversion, label encoding, MinMax normalization  
- 📉 **Dimensionality Reduction**: PCA for faster and cleaner model training  
- 🧾 **Streamlit Dashboard**: for monitoring attack statistics and model predictions  
- 💾 **Logging System**: stores attack logs and blocked IPs for later analysis  

---

## 🧩 Architecture
```text
+-------------------+
| Network Traffic   |
+---------+---------+
          |
          v
+-------------------+
| Data Preprocessor |
| (ARFF → Scaled → PCA) |
+---------+---------+
          |
          v
+-------------------+
| RL Agent (PPO)    |
| Learns to detect  |
| & mitigate attacks|
+---------+---------+
          |
          v
+-------------------+
| Streamlit Dashboard |
| Visualization + Logs |
+-------------------+
