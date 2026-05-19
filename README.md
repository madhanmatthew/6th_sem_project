# 🚑 AI Emergency Response & Smart Traffic Clearance System

> A simulation-driven intelligent emergency coordination platform designed to reduce ambulance response time and optimize emergency handling across Bengaluru using AI, Machine Learning, and Multi-Agent Systems.

---

## 🌟 Overview

The **AI Emergency Response & Smart Traffic Clearance System** is an intelligent smart-city simulation platform that automates the complete emergency dispatch workflow:

* 🚑 Ambulance dispatch
* 🏥 Trauma-aware hospital selection
* 🚦 Smart traffic signal clearance
* 👮 Police deployment coordination
* 📡 Real-time dispatch visualization

The system uses **five coordinated AI agents** working together through a **multi-agent coordinator** to simulate real-world emergency response optimization.

---

# ✨ Key Features

* 🚨 Real-time Emergency SOS Trigger
* 🧠 NLP-Based Severity Classification
* ⏱️ AI ETA Prediction System
* 🏥 Intelligent Hospital Optimization
* 🚦 Smart Green Corridor Simulation
* 👮 Automated Police Deployment
* 🗺️ Live Bengaluru Map Visualization
* 📊 Full Dispatch Analytics Dashboard
* 🔄 Multi-Agent Conflict Resolution
* 📡 WebSocket-Based Live Updates

---

# 🧠 AI Agents in the System

| Agent                       | Purpose                                                          |
| --------------------------- | ---------------------------------------------------------------- |
| NLP Severity Agent          | Classifies emergency severity using TF-IDF + Logistic Regression |
| ETA Prediction Agent        | Predicts ambulance arrival time using Gradient Boosting          |
| Hospital Optimization Agent | Selects the best trauma-capable hospital                         |
| Traffic Prediction Agent    | Chooses optimal signal corridor mode                             |
| Police Deployment Agent     | Allocates police units based on severity                         |

---

# 🏗️ System Architecture

The platform follows a **3-layer architecture**:

```text
Frontend Layer
    ↓
FastAPI Backend + Multi-Agent Coordinator
    ↓
Machine Learning Models & Data Layer
```

### Components

* **Frontend**

  * HTML5
  * Leaflet.js
  * OpenStreetMap

* **Backend**

  * FastAPI
  * REST APIs
  * WebSocket Streaming

* **Machine Learning**

  * scikit-learn
  * RandomForest
  * GradientBoosting
  * Logistic Regression

---

# 🛠️ Tech Stack

| Category        | Technologies                        |
| --------------- | ----------------------------------- |
| Frontend        | HTML5, CSS3, JavaScript, Leaflet.js |
| Backend         | FastAPI, Python                     |
| ML Libraries    | scikit-learn, pandas, numpy         |
| Mapping         | OpenStreetMap                       |
| Realtime        | WebSockets                          |
| Data            | GeoJSON, CSV                        |
| Version Control | Git, GitHub                         |

---

# 📂 Project Structure

```bash
AI-Emergency-Response-System/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── agents/
│   │   ├── ambulance_agent.py
│   │   ├── traffic_agent.py
│   │   ├── hospital_agent.py
│   │   ├── police_agent.py
│   │   └── coordinator.py
│   │
│   ├── ml/
│   │   ├── eta_model/
│   │   ├── severity_model/
│   │   ├── hospital_model/
│   │   └── traffic_model/
│   │
│   ├── data/
│   │   └── parser_utils.py
│   │
│   └── requirements.txt
│
├── frontend/
│   └── static/
│       ├── index.html
│       ├── map.html
│       ├── results.html
│       └── models.html
│
├── datasets/
│   ├── curated_emergency_hospitals.csv
│   ├── ambulance_bases.csv
│   └── bengaluru_hospitals.geojson
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ai-emergency-response-system.git
cd ai-emergency-response-system
```

---

## 2️⃣ Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

# ▶️ Run the Project

```bash
python api/main.py
```

Server will start at:

```text
http://localhost:8000
```

---

# 🔑 Optional OpenRouteService API Key

For advanced route generation:

## Linux / macOS

```bash
export ORS_API_KEY=your_openrouteservice_key
```

## Windows PowerShell

```powershell
$env:ORS_API_KEY = "your_openrouteservice_key"
```

---

# 🌐 Available Pages

| URL                   | Description                  |
| --------------------- | ---------------------------- |
| `/`                   | Landing Page                 |
| `/map`                | Live Emergency Map Dashboard |
| `/dispatch`           | Dispatch Console             |
| `/results`            | Dispatch Results Summary     |
| `/static/models.html` | ML Metrics Dashboard         |

---

# 🔌 API Endpoints

## POST `/dispatch`

Submit emergency report and receive AI decisions.

### Example Request

```json
{
  "location": "Silk Board",
  "incident": "Major accident",
  "injured": 5
}
```

---

## GET `/model-info`

Returns ML model metadata and evaluation metrics.

---

## GET `/ws/dispatch`

WebSocket endpoint for real-time dispatch workflow streaming.

---

# 🚨 Demo Flow

1. Open the **Live Map Dashboard**
2. Select:

   * Incident Location
   * Incident Type
   * Severity
   * Number of Injured
3. Press:

```text
TRIGGER EMERGENCY SOS
```

4. Watch the system:

   * 📍 Pin incident location
   * 🏥 Select best hospital
   * 🚑 Dispatch ambulance
   * 🚦 Activate green corridor
   * 👮 Deploy police
   * 📡 Show live notifications

---

# 📊 ML Models Used

| Model               | Algorithm                    | Purpose                      |
| ------------------- | ---------------------------- | ---------------------------- |
| Severity Classifier | TF-IDF + Logistic Regression | Emergency severity detection |
| ETA Predictor       | GradientBoostingRegressor    | Ambulance ETA prediction     |
| Hospital Optimizer  | RandomForest Regressor       | Trauma hospital selection    |
| Traffic Predictor   | RandomForest Classifier      | Congestion prediction        |

---

# 📍 Data Sources

| Dataset                           | Purpose                    |
| --------------------------------- | -------------------------- |
| `curated_emergency_hospitals.csv` | Bengaluru hospital data    |
| `ambulance_bases.csv`             | EMS base locations         |
| `bengaluru_hospitals.geojson`     | Hospital GPS coordinates   |
| OpenStreetMap                     | Live routing and map tiles |

---

# 📈 Benchmark Highlights

* ✅ Trauma-appropriate hospital selected in **5/8 scenarios**
* ✅ ETA RMSE: **1.28 minutes**
* ✅ Hospital Q-value RMSE: **0.037**
* ✅ Real Bengaluru emergency simulation support
* ✅ Multi-agent conflict resolution in all test scenarios

---

# 🔄 Multi-Agent Coordination

The coordinator resolves:

* 🚑 Ambulance vs Police route conflicts
* 🏥 Hospital bed availability conflicts
* 🚦 Traffic congestion rerouting
* ⚡ Priority-based emergency handling

---

# 🚀 Future Scope

* Live BBMP Hospital API Integration
* Real GPS Ambulance Tracking
* IoT Traffic Signal Integration
* CCTV-Based Accident Detection
* Reinforcement Learning Route Optimization
* Mobile Application Support
* Cloud Deployment on AWS
* Multi-City Karnataka Expansion

---

# 👨‍💻 Authors

* Madhan Matthew S
* Chetan Prasanna Sirsikar
* Raghavendra E S
* Sankalp L H

---

# 🏫 Institution

**CMR Institute of Technology**
Department of Artificial Intelligence & Machine Learning
Visvesvaraya Technological University (VTU)

---

# 📜 License

This project is developed for academic and research purposes.

---

# ⭐ Acknowledgment

Special thanks to:

* FastAPI
* scikit-learn
* OpenStreetMap
* Leaflet.js
* KASS Emergency Dataset

for enabling this smart-city emergency response simulation platform.

---


