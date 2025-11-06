# app_quantum_v2.py — Quantum 6G Security Simulator (Qiskit 1.4.5+)
import streamlit as st
import pandas as pd
import numpy as np
import hashlib, json, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Qiskit (modern v1.x imports) ---
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# --- Streamlit setup ---
st.set_page_config(page_title="Quantum 6G Security Simulator", layout="wide")
st.title("⚛️ Quantum-Enhanced 6G Security Simulator")
st.caption("Integrating THz-band 6G modeling, Quantum ML (QSVC), QKD, and Blockchain — powered by Qiskit 1.4.5+")

# --------------------------------------------------------------------
# 1️⃣ Simulating 6G Network Traffic
# --------------------------------------------------------------------
st.header("1️⃣ Simulating 6G Network Traffic (THz + AI-Driven Metrics)")

N = st.slider("Number of packets", 50, 300, 100)
attack_ratio = st.slider("Attack percentage", 5, 40, 15)

np.random.seed(42)

# --- Core 6G Parameters ---
freq_GHz = np.random.uniform(100, 1000, N)       # 100 GHz to 1 THz
bandwidth_GHz = np.random.uniform(1, 10, N)      # GHz range bandwidth
snr_db = np.random.normal(30, 5, N)              # Signal-to-noise ratio in dB

# --- Network Metrics ---
signal_strength = np.random.normal(70, 10, N)
latency = np.random.normal(1.5, 0.4, N)          # 6G: ultra-low latency
packet_loss = np.random.normal(0.3, 0.1, N)

# --- Attack Simulation ---
attack = np.random.choice([0, 1], size=N, p=[1-attack_ratio/100, attack_ratio/100])

latency[attack == 1] *= np.random.uniform(1.8, 2.5)
packet_loss[attack == 1] *= np.random.uniform(2.5, 4.0)
snr_db[attack == 1] -= np.random.uniform(8, 15)
signal_strength[attack == 1] -= np.random.uniform(10, 20)

# --- Compute 6G Shannon Data Rate ---
# C = B * log2(1 + SNR)
data_rate_Gbps = bandwidth_GHz * np.log2(1 + 10 ** (snr_db / 10)) / 1e3  # Gbps

# --- Combine into DataFrame ---
data = pd.DataFrame({
    "frequency_GHz": freq_GHz,
    "bandwidth_GHz": bandwidth_GHz,
    "SNR_dB": snr_db,
    "signal_strength": signal_strength,
    "latency": latency,
    "packet_loss": packet_loss,
    "data_rate_Gbps": data_rate_Gbps,
    "attack": attack
})
st.dataframe(data.head())

# --- Visualization of 6G Channel Behavior ---
fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(data["SNR_dB"], data["data_rate_Gbps"],
           c=data["attack"], cmap="coolwarm", alpha=0.7)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Data Rate (Gbps)")
ax.set_title("6G Channel Behavior (Red = Attack, Blue = Normal)")
st.pyplot(fig)

# --------------------------------------------------------------------
# 2️⃣ Quantum Machine Learning (QML)
# --------------------------------------------------------------------
st.header("2️⃣ Quantum Machine Learning Classification")

X = data[["signal_strength", "latency", "packet_loss", "SNR_dB", "data_rate_Gbps"]]
y = data["attack"]

st.warning("⚡ Quantum training in progress... Please wait 1-2 minutes")
with st.spinner("Training quantum QSVC model..."):
    X_scaled = (X - X.mean()) / X.std()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Quantum kernel + QSVC
    feature_map = ZZFeatureMap(feature_dimension=5, reps=1)
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    qml_model = QSVC(quantum_kernel=quantum_kernel)
    qml_model.fit(X_train, y_train)
    score = qml_model.score(X_test, y_test) * 100
    data["predicted_attack"] = qml_model.predict(X_scaled)

    st.success(f"✅ Quantum QSVC trained successfully — Test Accuracy: {score:.2f}%")

# --------------------------------------------------------------------
# 2D + 3D Visualization of 6G Quantum Security Patterns
# --------------------------------------------------------------------
import plotly.express as px

st.subheader("🌐 Visualizing Quantum 6G Security Landscape")

fig = px.scatter_3d(
    data,
    x="SNR_dB", y="latency", z="data_rate_Gbps",
    color=data["predicted_attack"].map({0: "Normal", 1: "Attack"}),
    size="packet_loss",
    hover_data=["frequency_GHz", "bandwidth_GHz", "signal_strength"],
    color_discrete_map={"Normal": "blue", "Attack": "red"},
    title="3D View — 6G Channel & Quantum Security Classification"
)
fig.update_layout(
    scene=dict(
        xaxis_title="SNR (dB)",
        yaxis_title="Latency (ms)",
        zaxis_title="Data Rate (Gbps)"
    ),
    legend_title_text="Prediction",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# 3️⃣ Quantum Key Distribution (QKD)
# --------------------------------------------------------------------
st.header("3️⃣ Quantum Key Distribution (QKD) Simulation")

def qkd_simulation(key_length=16):
    alice_bits = np.random.randint(2, size=key_length)
    alice_bases = np.random.randint(2, size=key_length)
    bob_bases = np.random.randint(2, size=key_length)
    bob_bits = np.array([
        alice_bits[i] if alice_bases[i] == bob_bases[i]
        else np.random.randint(2)
        for i in range(key_length)
    ])
    matching = np.where(alice_bases == bob_bases)[0]
    shared_key = ''.join(str(bob_bits[i]) for i in matching[:8])
    return shared_key, len(matching)

shared_key, matches = qkd_simulation()
st.info(f"🔑 QKD established shared key: **{shared_key}** (from {matches} matching bases)")
st.caption("In real QKD, eavesdropping changes qubit states — simulated conceptually here.")

# --------------------------------------------------------------------
# 4️⃣ Quantum Blockchain (QB)
# --------------------------------------------------------------------
st.header("4️⃣ Quantum Blockchain Ledger for 6G Security Events")

class Block:
    def __init__(self, index, timestamp, data, previous_hash=""):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calc_hash()
    def calc_hash(self):
        s = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash
        }, sort_keys=True).encode()
        return hashlib.sha256(s).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.genesis_block()]
    def genesis_block(self):
        return Block(0, time.time(), "Genesis Block", "0")
    def add_block(self, data):
        prev = self.chain[-1]
        new_block = Block(len(self.chain), time.time(), data, prev.hash)
        self.chain.append(new_block)

bc = Blockchain()
for i, row in data.iterrows():
    if row.predicted_attack == 1:
        bc.add_block({
            "id": int(i),
            "latency": float(row.latency),
            "packet_loss": float(row.packet_loss),
            "SNR_dB": float(row.SNR_dB),
            "qkd_key": shared_key,
            "verified": True
        })

st.success(f"🧱 Blockchain contains {len(bc.chain)} blocks (including Genesis).")
st.json({
    "index": bc.chain[-1].index,
    "hash": bc.chain[-1].hash[:40] + "...",
    "data": bc.chain[-1].data
})

# --------------------------------------------------------------------
# 5️⃣ Export Ledger
# --------------------------------------------------------------------
st.header("5️⃣ Export Blockchain Ledger")
ledger = [
    {"index": b.index, "timestamp": b.timestamp, "data": b.data,
     "hash": b.hash, "prev_hash": b.previous_hash}
    for b in bc.chain
]
json_bytes = json.dumps(ledger, indent=2).encode("utf-8")
st.download_button("⬇️ Download Blockchain Ledger (.json)",
                   data=json_bytes,
                   file_name="quantum_blockchain_ledger.json",
                   mime="application/json")

# --------------------------------------------------------------------
# 6️⃣ Summary
# --------------------------------------------------------------------
st.header("6️⃣ Summary")
st.markdown(f"""
### 🚀 System Overview
- **6G Traffic Model:** THz-band simulation with frequency, bandwidth, SNR, and data rate  
- **Quantum ML:** QSVC using FidelityQuantumKernel (StatevectorSampler backend)  
- **QKD:** Secure quantum key exchange simulation  
- **Quantum Blockchain:** Immutable ledger for verified 6G security events  
- **Accuracy:** {score:.2f}%  
- **Detected Attacks:** {sum(data.predicted_attack)} / {N}
""")
