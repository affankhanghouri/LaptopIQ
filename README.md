# 💻 LaptopIQ:  Price Intelligence System

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![AWS](https://img.shields.io/badge/Storage-AWS_S3-232F3E?style=flat-square&logo=amazons3&logoColor=white)](https://aws.amazon.com/s3/)
[![MLOps](https://img.shields.io/badge/Workflow-MLOps-FF6F00?style=flat-square&logo=git&logoColor=white)]()

**LaptopIQ** is not just a model; it is a high-performance, engineering-first MLOps ecosystem designed to bridge the gap between experimental machine learning and production-grade software. It automates the entire lifecycle of laptop price prediction—from raw data ingestion in MongoDB to real-time inference via a high-performance FastAPI backend.



---
![System Architecture](https://github.com/user-attachments/assets/2e832e18-4223-469d-a4db-456d41bcb514)

##  Project frontend :



<img width="445" height="552" alt="Image" src="https://github.com/user-attachments/assets/c5447a73-ee67-4607-a744-2df249b78b2d" />




## 🏗️ System Architecture & Engineering Stack

The system is architected as a decoupled, artifact-driven pipeline. This ensures that the data science workflow is reproducible, auditable, and ready for CI/CD integration.

### High-Level Workflow
1.  **Persistence Layer:** Raw product data stored and versioned in **MongoDB**.
2.  **Orchestrated Pipeline:** Modular stages for Ingestion, Validation, and Transformation.
3.  **Model Registry:** Trained models and pre-processing artifacts are versioned and stored in **AWS S3**.
4.  **Serving Layer:** An asynchronous **FastAPI** service provides low-latency inference.
5.  **Reactive UI:** A dedicated frontend provides an intuitive interface for end-user price appraisals.

---




# 📂 Project Structure

```text
LAPTOPIQ/
├── .github/workflows/          # CI/CD pipelines (Automated Testing/Deployment)
├── artifact/                   # Local versioned pipeline artifacts (Model/Data)
├── config/                     # YAML environment & schema definitions
├── logs/                       # Centralized application & pipeline logs
├── notebook_experiments/       # R&D and sandbox experimentation
│
├── src/                        # Core Production Engine
│   ├── cloud_storage/          # AWS S3 Integration & Model Registry logic
│   ├── data_access/            # MongoDB persistence & abstract DAO layer
│   ├── components/             # Modular Pipeline Units
│   │   ├── data_ingestion/     # Automated train/test splitting & sourcing
│   │   ├── data_validation/    # Strict schema & drift enforcement
│   │   ├── data_transformation/# Feature engineering & preprocessing
│   │   ├── model_trainer/      # Hyperparameter-tuned RandomForest Training
│   │   ├── model_evaluation/   # Model comparison & performance metrics
│   │   └── model_pusher/       # Production deployment to S3 Registry
│   ├── pipeline/               # Orchestration of Training & Prediction flows
│   ├── entity/                 # Configuration & Artifact entities
│   ├── constants/              # System-wide immutable constants
│   ├── utils/                  # Common helper functions & I/O operations
│   ├── logger/                 # Custom industry-standard logging
│   └── exception/              # Global custom exception handling framework
│
├── templates/                  # Frontend UI templates (HTML/Jinja2)
├── static/                     # CSS, JS, and high-fidelity assets
├── main.py                     # FastAPI Application Entry Point
└── requirements.txt            # Dependency Management



```

## 🧠 Core Engineering Principles

### 1. Robust Production Serving (FastAPI)
Unlike standard Flask apps, LaptopIQ utilizes **FastAPI** to handle asynchronous requests, ensuring the system can scale under high concurrency.


### 2. Schema-First Data Validation
To prevent "garbage-in, garbage-out," the system implements a strict validation layer. Every data batch is checked against a predefined schema before entering the transformation pipeline, ensuring 100% data integrity.

### 3. Decoupled Transformation Logic
We treat feature engineering as a standalone service. By separating transformation logic from training code, we ensure that the exact same preprocessing "pickles" used during training are applied during real-time inference, eliminating **Training-Serving Skew**.

### 4. Artifact-Based Traceability
Every run generates a unique set of artifacts:
* **Data Artifacts:** Train/Test splits and transformed feature sets.
* **Model Artifacts:** Serialized models, encoders, and scalers.
* **Logs:** Centralized logging for debugging pipeline failures in production.

---


## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend** | FastAPI | High-performance asynchronous API serving |
| **Frontend** | Streamlit/React | User interface for real-time predictions |
| **Database** | MongoDB | NoSQL storage for flexible laptop specifications |
| **Cloud Storage** | AWS S3 | Centralized Model Registry and Artifact Store |
| **ML Framework** | Scikit-Learn / RandomForestRegressor | RFG for price estimation |


---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* MongoDB Instance (Local or Atlas)
* AWS Credentials (for S3 Artifact Access)

### Installation
```bash
# Clone the repository
git clone [https://github.com/yourusername/LaptopIQ.git](https://github.com/yourusername/LaptopIQ.git)

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI Server
uvicorn app:app --reload
