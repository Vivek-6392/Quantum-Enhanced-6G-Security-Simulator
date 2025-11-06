# # app_quantum_v2.py — Quantum 6G Security Simulator
# import streamlit as st
# import pandas as pd
# import numpy as np
# import hashlib, json, time
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# # --- Qiskit (modern v1.x imports) ---
# try:
#     from qiskit.circuit.library import ZZFeatureMap
#     from qiskit.primitives import StatevectorSampler
#     from qiskit_algorithms.state_fidelities import ComputeUncompute
#     from qiskit_machine_learning.kernels import FidelityQuantumKernel
#     from qiskit_machine_learning.algorithms import QSVC
#     QISKIT_AVAILABLE = True
# except ImportError as e:
#     QISKIT_AVAILABLE = False
#     st.error(f"⚠️ Qiskit import failed: {e}")

# # --- Streamlit setup ---
# st.set_page_config(page_title="Quantum 6G Security Simulator", layout="wide")
# st.title("⚛️ Quantum-Enhanced 6G Security Simulator (Modern Qiskit API)")
# st.caption("Using FidelityQuantumKernel + QSVC — full quantum simulation with Qiskit 1.4.5")

# # --------------------------------------------------------------------
# # 1️⃣ Simulate 6G Network Traffic
# # --------------------------------------------------------------------
# st.header("1️⃣ Simulating 6G Network Traffic")
# N = st.slider("Number of packets", 50, 300, 150)  # Reduced max for performance
# attack_ratio = st.slider("Attack percentage", 5, 40, 15)

# np.random.seed(42)
# data = pd.DataFrame({
#     "signal_strength": np.random.normal(70, 10, N),
#     "latency": np.random.normal(2, 0.5, N),
#     "packet_loss": np.random.normal(0.5, 0.2, N),
#     "attack": np.random.choice([0, 1], size=N, p=[1-attack_ratio/100, attack_ratio/100])
# })
# data.loc[data.attack == 1, "latency"] *= 2
# data.loc[data.attack == 1, "packet_loss"] *= 3
# st.dataframe(data.head(10))
# st.info(f"📊 Generated {N} packets ({sum(data.attack)} attacks, {N - sum(data.attack)} normal)")

# # --------------------------------------------------------------------
# # 2️⃣ Quantum Machine Learning (QML)
# # --------------------------------------------------------------------
# st.header("2️⃣ Quantum Machine Learning Classification")

# if not QISKIT_AVAILABLE:
#     st.error("❌ Qiskit not available. Install with: `pip install qiskit qiskit-machine-learning`")
#     st.stop()

# X = data[["signal_strength", "latency", "packet_loss"]]
# y = data["attack"]
# X_scaled = (X - X.mean()) / X.std()

# # **KEY FIX: Reduce training size dramatically**
# sample_size = min(50, len(X_scaled))  # Use only 50 samples for training
# X_sample = X_scaled.sample(n=sample_size, random_state=42)
# y_sample = y.loc[X_sample.index]
# X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

# st.warning(f"⚡ Training on {len(X_train)} samples (quantum kernels are computationally expensive)")

# # Progress indicator
# with st.spinner("🔄 Training Quantum QSVC... This may take 30-60 seconds..."):
#     try:
#         # Feature map & quantum components
#         feature_map = ZZFeatureMap(feature_dimension=3, reps=1)
#         sampler = StatevectorSampler()
#         fidelity = ComputeUncompute(sampler=sampler)
#         quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
        
#         qml_model = QSVC(quantum_kernel=quantum_kernel)
#         qml_model.fit(X_train.values, y_train.values)
        
#         # Predict on full dataset
#         data["predicted_attack"] = qml_model.predict(X_scaled.values)
#         score = qml_model.score(X_test.values, y_test.values) * 100
        
#         st.success(f"✅ Quantum QSVC trained successfully — test accuracy: {score:.2f}%")
        
#     except Exception as e:
#         st.error(f"❌ Quantum training failed: {e}")
#         st.info("Using classical fallback (random predictions for demo)")
#         data["predicted_attack"] = np.random.choice([0, 1], size=len(data))
#         score = 50.0

# # Visualization
# fig, ax = plt.subplots(figsize=(10, 6))
# scatter = ax.scatter(data["latency"], data["packet_loss"],
#            c=data["predicted_attack"], cmap="coolwarm", alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
# ax.set_xlabel("Latency (ms)", fontsize=12)
# ax.set_ylabel("Packet Loss (%)", fontsize=12)
# ax.set_title("Quantum ML Predicted Behavior (Red = Attack, Blue = Normal)", fontsize=14)
# plt.colorbar(scatter, ax=ax, label="Predicted Attack")
# st.pyplot(fig)

# # --------------------------------------------------------------------
# # 3️⃣ Quantum Key Distribution (QKD)
# # --------------------------------------------------------------------
# st.header("3️⃣ Quantum Key Distribution (QKD) Simulation")

# def qkd_simulation(key_length=16):
#     alice_bits = np.random.randint(2, size=key_length)
#     alice_bases = np.random.randint(2, size=key_length)
#     bob_bases = np.random.randint(2, size=key_length)
#     bob_bits = np.array([
#         alice_bits[i] if alice_bases[i] == bob_bases[i]
#         else np.random.randint(2)
#         for i in range(key_length)
#     ])
#     matching = np.where(alice_bases == bob_bases)[0]
#     shared_key = ''.join(str(bob_bits[i]) for i in matching[:8])
#     return shared_key, len(matching)

# shared_key, matches = qkd_simulation()
# st.info(f"🔑 QKD established shared key: **{shared_key}** (from {matches} matching bases)")
# st.caption("In real QKD, eavesdropping changes qubit states — simulated here conceptually.")

# # --------------------------------------------------------------------
# # 4️⃣ Quantum Blockchain (QB)
# # --------------------------------------------------------------------
# st.header("4️⃣ Quantum Blockchain Ledger")

# class Block:
#     def __init__(self, index, timestamp, data, previous_hash=""):
#         self.index = index
#         self.timestamp = timestamp
#         self.data = data
#         self.previous_hash = previous_hash
#         self.hash = self.calc_hash()
#     def calc_hash(self):
#         s = json.dumps({
#             "index": self.index,
#             "timestamp": self.timestamp,
#             "data": self.data,
#             "previous_hash": self.previous_hash
#         }, sort_keys=True).encode()
#         return hashlib.sha256(s).hexdigest()

# class Blockchain:
#     def __init__(self):
#         self.chain = [self.genesis_block()]
#     def genesis_block(self):
#         return Block(0, time.time(), "Genesis Block", "0")
#     def add_block(self, data):
#         prev = self.chain[-1]
#         new_block = Block(len(self.chain), time.time(), data, prev.hash)
#         self.chain.append(new_block)

# with st.spinner("🔄 Building blockchain..."):
#     bc = Blockchain()
#     for i, row in data.iterrows():
#         if row.predicted_attack == 1:
#             bc.add_block({
#                 "id": int(i),
#                 "latency": float(row.latency),
#                 "packet_loss": float(row.packet_loss),
#                 "qkd_key": shared_key,
#                 "verified": True
#             })

# st.success(f"🧱 Blockchain contains {len(bc.chain)} blocks (including Genesis).")
# if len(bc.chain) > 1:
#     st.json({
#         "index": bc.chain[-1].index,
#         "hash": bc.chain[-1].hash[:40] + "...",
#         "data": bc.chain[-1].data
#     })

# # --------------------------------------------------------------------
# # 5️⃣ Export Ledger
# # --------------------------------------------------------------------
# st.header("5️⃣ Export Blockchain Ledger")
# ledger = [
#     {"index": b.index, "timestamp": b.timestamp, "data": b.data,
#      "hash": b.hash, "prev_hash": b.previous_hash}
#     for b in bc.chain
# ]
# json_bytes = json.dumps(ledger, indent=2).encode("utf-8")
# st.download_button("⬇️ Download Blockchain Ledger (.json)",
#                    data=json_bytes,
#                    file_name="quantum_blockchain_ledger.json",
#                    mime="application/json")

# # --------------------------------------------------------------------
# # 6️⃣ Summary
# # --------------------------------------------------------------------
# st.header("6️⃣ Summary")
# st.markdown(f"""
# - **Quantum ML:** QSVC using FidelityQuantumKernel (StatevectorSampler)  
# - **QKD:** Simulated secure key exchange  
# - **Quantum Blockchain:** Immutable ledger storing verified attacks  
# - **Accuracy:** {score:.2f}%  |  **Detected Attacks:** {sum(data.predicted_attack)} / {N}  
# - **Training samples:** {len(X_train)} (reduced for performance)
# """)

# st.info("💡 **Tip:** Reduce packet count for faster quantum computations")

# app_quantum_v2_ui.py — Quantum 6G Security Simulator (Enhanced UI)
import streamlit as st
import pandas as pd
import numpy as np
import hashlib, json, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Qiskit imports ---
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    st.error(f"⚠️ Qiskit import failed: {e}")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Quantum 6G Security Simulator", layout="wide", page_icon="⚛️")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    /* Background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f2f2f2;
    }

    [data-testid="stHeader"] {background: rgba(0,0,0,0);}

    /* Headings */
    h1, h2, h3, h4 {
        color: #00e0ff !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Cards */
    .stContainer {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }

    /* Buttons */
    div.stButton > button {
        background-color: #00e0ff;
        color: #000;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.title("⚛️ Quantum-Enhanced 6G Security Simulator")
st.caption("Built with FidelityQuantumKernel + QSVC — powered by Qiskit 1.4.5")

# --------------------------------------------------------------------
# 1️⃣ Simulating 6G Network Traffic
# --------------------------------------------------------------------
st.markdown("## 🌐 Step 1: Simulate 6G Network Traffic")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        N = st.slider("Number of packets", 50, 300, 150)
    with col2:
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

    st.dataframe(data.head(10), use_container_width=True)
    st.success(f"📊 Generated {N} packets — {sum(data.attack)} attacks detected.")

# --------------------------------------------------------------------
# 2️⃣ Quantum Machine Learning
# --------------------------------------------------------------------
st.markdown("## 🧠 Step 2: Quantum Machine Learning (QML) Classification")

if not QISKIT_AVAILABLE:
    st.error("❌ Qiskit not available. Run `pip install qiskit qiskit-machine-learning`.")
    st.stop()

X = data[["signal_strength", "latency", "packet_loss"]]
y = data["attack"]
X_scaled = (X - X.mean()) / X.std()

sample_size = min(50, len(X_scaled))
X_sample = X_scaled.sample(n=sample_size, random_state=42)
y_sample = y.loc[X_sample.index]
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

st.warning(f"⚡ Training Quantum QSVC on {len(X_train)} samples (reduced for speed)")

with st.spinner("🔄 Training Quantum QSVC (using FidelityQuantumKernel)..."):
    try:
        feature_map = ZZFeatureMap(feature_dimension=3, reps=1)
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

        qml_model = QSVC(quantum_kernel=quantum_kernel)
        qml_model.fit(X_train.values, y_train.values)

        data["predicted_attack"] = qml_model.predict(X_scaled.values)
        score = qml_model.score(X_test.values, y_test.values) * 100

        st.success(f"✅ Quantum QSVC trained — Accuracy: **{score:.2f}%**")
    except Exception as e:
        st.error(f"❌ Quantum training failed: {e}")
        data["predicted_attack"] = np.random.choice([0, 1], size=len(data))
        score = 50.0

# --- Visualization (smaller & clean) ---
fig, ax = plt.subplots(figsize=(5, 3))
scatter = ax.scatter(
    data["latency"], data["packet_loss"],
    c=data["predicted_attack"], cmap="coolwarm", alpha=0.8,
    s=50, edgecolors="black", linewidth=0.4
)
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Packet Loss (%)")
ax.set_title("Quantum ML Predictions", fontsize=12)
plt.colorbar(scatter, ax=ax, label="Attack (1) / Normal (0)")
st.pyplot(fig)

# --------------------------------------------------------------------
# 3️⃣ Quantum Key Distribution (QKD)
# --------------------------------------------------------------------
st.markdown("## 🔐 Step 3: Quantum Key Distribution (QKD) Simulation")

def qkd_simulation(key_length=16):
    alice_bits = np.random.randint(2, size=key_length)
    alice_bases = np.random.randint(2, size=key_length)
    bob_bases = np.random.randint(2, size=key_length)
    bob_bits = np.array([
        alice_bits[i] if alice_bases[i] == bob_bases[i] else np.random.randint(2)
        for i in range(key_length)
    ])
    matching = np.where(alice_bases == bob_bases)[0]
    shared_key = ''.join(str(bob_bits[i]) for i in matching[:8])
    return shared_key, len(matching)

shared_key, matches = qkd_simulation()
st.info(f"🔑 Shared QKD Key: **{shared_key}** (Matched Bases: {matches})")

# --------------------------------------------------------------------
# 4️⃣ Quantum Blockchain
# --------------------------------------------------------------------
st.markdown("## ⛓️ Step 4: Quantum Blockchain Ledger")

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

with st.spinner("🔄 Building blockchain..."):
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

st.success(f"🧱 Blockchain created — {len(bc.chain)} blocks (Genesis + {len(bc.chain)-1} attacks)")
if len(bc.chain) > 1:
    st.json({
        "index": bc.chain[-1].index,
        "hash": bc.chain[-1].hash[:40] + "...",
        "data": bc.chain[-1].data
    })

# --------------------------------------------------------------------
# 5️⃣ Export Blockchain Ledger
# --------------------------------------------------------------------
st.markdown("## 💾 Step 5: Export Blockchain Ledger")

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
st.markdown("## 🧩 Step 6: Summary")
st.markdown(f"""
| Component | Description |
|------------|-------------|
| 🧠 **Quantum ML** | QSVC using FidelityQuantumKernel |
| 🔑 **QKD** | Simulated BB84-style key exchange |
| ⛓️ **Quantum Blockchain** | Stores verified attack data |
| 📈 **Accuracy** | {score:.2f}% |
| ⚙️ **Detected Attacks** | {sum(data.predicted_attack)} / {N} |
| 🧩 **Training Samples** | {len(X_train)} (optimized for performance) |
""", unsafe_allow_html=True)

st.info("💡 Tip: Reduce packet count for faster quantum simulation.")
