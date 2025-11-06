# app_quantum_v2.py — Quantum 6G Security Simulator (for Qiskit 1.4.5+)
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
st.title("⚛️ Quantum-Enhanced 6G Security Simulator (Modern Qiskit API)")
st.caption("Using FidelityQuantumKernel + QSVC — full quantum simulation with Qiskit 1.4.5")

# --------------------------------------------------------------------
# 1️⃣ Simulate 6G Network Traffic
# --------------------------------------------------------------------
st.header("1️⃣ Simulating 6G Network Traffic")
N = st.slider("Number of packets", 50, 300, 50)
attack_ratio = st.slider("Attack percentage", 5, 40, 15)

np.random.seed(42)
data = pd.DataFrame({
    "signal_strength": np.random.normal(70, 10, N),
    "latency": np.random.normal(2, 0.5, N),
    "packet_loss": np.random.normal(0.5, 0.2, N),
    "attack": np.random.choice([0, 1], size=N, p=[1-attack_ratio/100, attack_ratio/100])
})
data.loc[data.attack == 1, "latency"] *= 2
data.loc[data.attack == 1, "packet_loss"] *= 3
st.dataframe(data.head())

# --------------------------------------------------------------------
# 2️⃣ Quantum Machine Learning (QML)
# --------------------------------------------------------------------
st.header("2️⃣ Quantum Machine Learning Classification")

X = data[["signal_strength", "latency", "packet_loss"]]
# Add this before line "X = data[...]"
st.warning("⚡ Quantum training in progress... Please wait 1-2 minutes")
with st.spinner("Training quantum model..."):
    # ... your existing QML code here
    y = data["attack"]
    X_scaled = (X - X.mean()) / X.std()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Feature map & quantum components
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1)
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    qml_model = QSVC(quantum_kernel=quantum_kernel)
    qml_model.fit(X_train, y_train)
    score = qml_model.score(X_test, y_test) * 100
    data["predicted_attack"] = qml_model.predict(X_scaled)

    st.success(f"✅ Quantum QSVC trained successfully — test accuracy: {score:.2f}%")

# Visualization
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(data["latency"], data["packet_loss"],
           c=data["predicted_attack"], cmap="coolwarm", alpha=0.8)
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Packet Loss (%)")
ax.set_title("Quantum ML Predicted Behavior (Red = Attack, Blue = Normal)")
st.pyplot(fig)

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
st.caption("In real QKD, eavesdropping changes qubit states — simulated here conceptually.")

# --------------------------------------------------------------------
# 4️⃣ Quantum Blockchain (QB)
# --------------------------------------------------------------------
st.header("4️⃣ Quantum Blockchain Ledger")

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
- **Quantum ML:** QSVC using FidelityQuantumKernel (StatevectorSampler)  
- **QKD:** Simulated secure key exchange  
- **Quantum Blockchain:** Immutable ledger storing verified attacks  
- **Accuracy:** {score:.2f}%  |  **Detected Attacks:** {sum(data.predicted_attack)} / {N}
""")
