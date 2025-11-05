# ⚛️ Quantum-Enhanced 6G Security Simulator  

### Developed by  
**Vivek Yadav (BT23CSA035)**  
**Indian Institute of Information Technology, Nagpur**

---

## 🧠 Abstract  

The **Quantum-Enhanced 6G Security Simulator** integrates three cutting-edge technologies — **Quantum Machine Learning (QML)**, **Quantum Key Distribution (QKD)**, and **Quantum Blockchain (QB)** — to model a secure and intelligent 6G communication system.  

- **Quantum Machine Learning (QML):** Uses QSVC (Quantum Support Vector Classifier) with Fidelity Quantum Kernel for classifying network traffic and detecting malicious behavior.  
- **Quantum Key Distribution (QKD):** Simulates quantum-secure key exchange, ensuring tamper-proof encryption.  
- **Quantum Blockchain (QB):** Maintains a tamper-proof distributed ledger to store detected attacks verified through QKD keys.

This project demonstrates how **quantum computing** can improve **network security and reliability** in next-generation 6G systems.

---

## ⚙️ Features  

✅ Simulates realistic 6G traffic (signal strength, latency, packet loss).  
✅ Detects network anomalies using **Quantum Machine Learning (QSVC)**.  
✅ Implements **Quantum Key Distribution (QKD)** for secure encryption key sharing.  
✅ Records verified attack data in a **Quantum Blockchain ledger**.  
✅ Exports the blockchain ledger as a `.json` file for external analysis.  

---

## 🧰 Tech Stack  

| Component | Technology Used |
|------------|----------------|
| Frontend | Streamlit |
| Quantum Framework | Qiskit 1.4.5 |
| Quantum ML | Qiskit Machine Learning 0.8.4 |
| Backend Simulation | Python (NumPy, Pandas, Matplotlib, Scikit-learn) |
| Environment | Python 3.13 |
| Platform | Streamlit UI / GitHub / Local Environment |

---

## 🧩 System Architecture  
```text
  ┌─────────────────────┐
  │   6G Traffic Data    │
  └────────┬────────────┘
           │
           ▼
┌───────────────────────────┐
│ Quantum Machine Learning  │
│ (QSVC + Fidelity Kernel)  │
└────────┬──────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │ Quantum Key          │
  │ Distribution (QKD)   │
  └────────┬────────────┘
           │
           ▼
┌──────────────────────────┐
│ Quantum Blockchain Ledger│
└──────────────────────────┘


```

## 🧾 Installation and Setup  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Vivek-6392/Quantum-Enhanced-6G-Security-Simulator.git
cd Quantum-Enhanced-6G-Security-Simulator
```
pip install -r requirements.txt

streamlit run app_quantum_v2.py
