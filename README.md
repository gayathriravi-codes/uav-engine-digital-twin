# Nidan

### AI-Enabled Digital Twin for UAV Engine Health Monitoring

**Nidan** is an AI-enabled digital twin prototype designed for **health monitoring, telemetry analysis, anomaly identification, and reliability assessment of UAV piston engines**.

The project combines engine telemetry, machine-learning workflows, and an interactive web-based digital-twin interface to provide a unified environment for monitoring engine behavior and visualizing operational insights.

> **Project Status:** Active Development

🌐 **Live Demo:** https://nidan-dun.vercel.app/

---

## Overview

UAV engines operate under changing environmental and operational conditions, making continuous monitoring important for identifying abnormal behavior and supporting maintenance decisions.

Nidan explores how a **digital-twin-based architecture** can combine telemetry data and AI/ML techniques to create a digital representation of an engine's operating condition.

The current prototype focuses on:

* Engine telemetry visualization
* Operational parameter monitoring
* Anomaly identification
* Engine health assessment
* Historical/replay analysis
* What-if exploration
* Near-real-time simulated monitoring
* Interactive digital-twin visualization

---

## Problem Statement

Traditional engine monitoring systems often involve large volumes of sensor data that require manual interpretation and separate analysis pipelines.

Nidan aims to provide a unified software environment where telemetry can be processed, visualized, and analyzed to help identify abnormal operating patterns and support engine health monitoring.

### Core Question

> **Can AI/ML-driven telemetry analysis and a digital-twin interface provide a more integrated approach to monitoring the health and operational behavior of UAV piston engines?**

---

## Objectives

1. Build a digital representation of a UAV piston engine using telemetry data.
2. Process and analyze engine operating parameters.
3. Identify abnormal or potentially anomalous operating behavior.
4. Provide an interactive monitoring interface for operators.
5. Support historical replay and what-if analysis.
6. Create a foundation for future predictive-maintenance and reliability research.

---

## System Architecture

```text
                    UAV ENGINE
                        │
                        ▼
               Engine Telemetry
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
              Feature Engineering
                        │
                        ▼
                 AI / ML Layer
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Health Analysis       Anomaly Analysis
             │                     │
             └──────────┬──────────┘
                        ▼
                Digital Twin Layer
                        │
                        ▼
              Nidan Web Interface
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
     Dashboard      Live Monitor      What-If
        │               │                │
        └───────────────┼────────────────┘
                        ▼
                 Operator Insights
```

---

## Key Features

### 📊 Engine Health Dashboard

Provides an overview of engine operating conditions and relevant telemetry indicators through an interactive interface.

### 📡 Live Monitoring

Displays a **near-real-time simulated telemetry stream** for demonstrating how engine parameters can be monitored continuously.

### 🔍 Anomaly Identification

The system is designed to analyze telemetry behavior and identify patterns that may indicate abnormal engine operation.

### 🔄 Replay Analysis

Historical telemetry can be replayed to examine changes in engine behavior over time.

### 🧪 What-If Analysis

Provides an environment for exploring how changes in operating conditions can affect monitored engine parameters.

### 🔌 API Integration

The frontend communicates with a backend API for retrieving and processing monitoring data.

### 🖥️ Interactive Web Interface

The Nidan interface provides a centralized environment for visualizing engine information and monitoring system outputs.

---

## Technology Stack

### Frontend

* React
* Vite
* JavaScript
* HTML/CSS

### Backend

* Python
* REST API
* FastAPI/Flask-based backend components

### AI / Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Machine Learning workflows

### Deployment

* **Frontend:** Vercel
* **Backend:** Render
* **Version Control:** Git & GitHub

---

## Project Structure

```text
uav-engine-digital-twin/
│
├── aerotwin-frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── ...
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── backend/
│   └── ...
│
├── README.md
└── ...
```

> The repository structure may evolve as the project moves through development.

---

## Data Pipeline

The general data-processing workflow is:

```text
Raw Engine Telemetry
        │
        ▼
Data Cleaning
        │
        ▼
Missing / Invalid Value Handling
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Machine Learning
        │
        ▼
Health / Anomaly Analysis
        │
        ▼
Visualization
```

---

## Current Implementation

The current Nidan prototype includes:

* Web-based engine monitoring interface
* Dashboard-based telemetry visualization
* Live simulated telemetry stream
* Historical/replay functionality
* What-if interface
* Backend API integration
* Deployment through Vercel and Render
* Engine telemetry datasets and preprocessing workflows
* Initial AI/ML analysis pipeline

The deployed demonstration currently uses **simulated telemetry for live monitoring**. The backend architecture is being developed toward integration with trained ML models and more advanced engine-health inference.

---

## Deployment

### Frontend

Nidan's frontend is deployed using **Vercel**.

🌐 **Live Demo:**
https://nidan-dun.vercel.app/

### Backend

The backend API is deployed separately using **Render**.

The frontend communicates with the backend through an environment-configured API endpoint.

---

## Research Direction

Nidan is being developed as more than a dashboard. The long-term research direction is to investigate AI-driven engine health monitoring under realistic operational conditions.

Potential research extensions include:

* Uncertainty-aware health estimation
* Robust anomaly detection
* Noisy and incomplete sensor data
* Sensor failure and missing telemetry
* Predictive maintenance
* Remaining Useful Life estimation
* Physics-informed machine learning
* Simulation-to-real transfer
* Explainable AI for engine health decisions
* Reliability-aware decision support

These components represent **future research directions and are not claimed as completed functionality in the current prototype**.

---

## Why Nidan?

Nidan focuses on the intersection of:

**Artificial Intelligence + Digital Twins + UAV Systems + Predictive Maintenance**

Rather than treating machine-learning predictions as isolated outputs, the project aims to place AI-based analysis inside a digital-twin environment where engine behavior can be monitored, interpreted, and eventually used to support maintenance and operational decisions.

---

## Future Work

* Integrate trained ML models directly into the production API
* Improve anomaly detection performance
* Incorporate uncertainty estimation
* Introduce realistic streaming telemetry
* Model sensor degradation and missing-data scenarios
* Develop predictive health indicators
* Explore RUL estimation
* Integrate physics-based constraints
* Evaluate the system using real-world or high-fidelity simulated UAV engine data
* Conduct systematic experimental evaluation

---

## Installation

### Clone the repository

```bash
git clone https://github.com/gayathriravi-codes/uav-engine-digital-twin.git

cd uav-engine-digital-twin
```

### Frontend

```bash
cd aerotwin-frontend

npm install

npm run dev
```

The Vite development server will start locally.

### Build for production

```bash
npm run build
```

The production build is generated in the `dist/` directory.

---

## Environment Configuration

The frontend uses an environment variable to connect to the backend API.

Example:

```env
VITE_API_URL=<your-backend-api-url>
```

For the deployed Nidan system, the frontend is configured to communicate with the deployed backend service.

---

## Project Goals

Nidan aims to evolve toward a research-oriented digital-twin framework capable of:

```text
Observe
   ↓
Analyze
   ↓
Detect
   ↓
Estimate
   ↓
Explain
   ↓
Support Decisions
```

The ultimate objective is to explore how AI can make UAV engine monitoring more **adaptive, interpretable, and reliable** under changing operating conditions.

---

## Project Status

🚧 **Active Development**

Nidan is currently a working prototype. The software architecture, AI/ML pipeline, datasets, and digital-twin capabilities are being continuously improved.

---

## Contributors

Developed as a collaborative AI & Data Science project.

**GitHub:**
https://github.com/gayathriravi-codes/uav-engine-digital-twin

---

## License

This project is currently intended for **academic and research purposes**.

A formal open-source license can be added as the project reaches a stable release.
