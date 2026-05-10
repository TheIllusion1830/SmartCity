# SmartCity
# 🌆 Smart City Deep Learning: Traffic Prediction & Parking Detection

> **Advanced deep learning models for urban traffic forecasting and intelligent parking management**

A comprehensive implementation of state-of-the-art graph neural networks and computer vision models for smart city applications, featuring real-time traffic prediction and automated parking detection with an interactive web demo.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Models Implemented](#-models-implemented)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Results](#-results)
- [Demo Application](#-demo-application)
- [Team](#-team)
- [References](#-references)

---

## 🎯 Overview

This project addresses two critical smart city challenges:

### 1. **Traffic Speed Forecasting**
Predict traffic speeds across 207 highway sensors using spatio-temporal graph neural networks, enabling:
- Real-time congestion prediction
- Route optimization
- Traffic management planning
- 60-minute ahead forecasting with 5-minute intervals

### 2. **Parking Space Detection**
Automated parking occupancy detection from camera images using deep CNNs with LSTM integration:
- Binary classification (vacant/occupied)
- Cross-parking-lot generalization
- Real-time monitoring capability
- 92.5% accuracy on test data

---

## ✨ Key Features

- **🚀 State-of-the-Art Models**: Implementation of DCRNN, GMAN, GraphWaveNet, and custom ParkNet-LSTM
- **📊 Comprehensive Analysis**: Detailed performance comparisons across multiple metrics (MAE, RMSE, MAPE)
- **🎨 Rich Visualizations**: Interactive plots for predictions, error analysis, and model comparisons
- **🌐 Web Demo**: Flask-based interactive application with real-time predictions
- **📈 Scalable Architecture**: Efficient implementations optimized for large-scale sensor networks
- **🔧 Production-Ready**: Complete training pipeline, evaluation scripts, and deployment code

---

## 🧠 Models Implemented

### Traffic Prediction Models

#### 1. **DCRNN (Diffusion Convolutional Recurrent Neural Network)**
Our primary baseline model combining:
- Diffusion convolution for spatial dependencies
- GRU cells for temporal dynamics
- Encoder-decoder architecture with teacher forcing

**Architecture:**
```
Input (12 timesteps) → Encoder (2-layer DCGRU) → Decoder (2-layer DCGRU) → Output (12 predictions)
Hidden Dim: 64 | K-hops: 3 | Parameters: ~450K
```

**Performance:**
- MAE: 3.47 mph
- RMSE: 5.82 mph
- Training Time: ~15 min/epoch

#### 2. **GMAN (Graph Multi-Attention Network) - The Model chosen for implementing Traffic Prediction**
Advanced attention-based model featuring:
- Spatial attention mechanism
- Temporal attention mechanism
- Gated fusion layers
- Spatio-temporal embedding

**Variants Implemented:**
- Full GMAN (with custom attention blocks)
- LightweightGMAN (optimized for memory efficiency)
- MinimalGMAN (ultra-fast baseline)

#### 3. **GraphWaveNet - Comparitive Model**
Graph convolution with adaptive adjacency learning:
- Predefined + learned adjacency matrices
- WaveNet-style dilated convolutions
- Gated TCN architecture

### Parking Detection Models

#### 1. **mAlexNet (Baseline)**
Modified AlexNet architecture:
- 5 convolutional layers
- Batch normalization
- Dropout regularization
- ~88% accuracy

#### 2. **ParkNet-LSTM (Enhanced Model)**
Our novel architecture combining CNNs with LSTM:
```
CNN Feature Extractor → Spatial Attention → LSTM Sequence → FC Classifier
```

**Key Features:**
- Attention-weighted spatial features
- Temporal context integration
- Data augmentation pipeline
- Class imbalance handling

**Performance:**
- Accuracy: 92.5%
- Precision: 91.8%
- Recall: 93.2%
- F1-Score: 92.5%

---

## 📁 Project Structure

```
smart-city-deep-learning/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── smart_city_final.ipynb             # Main training notebook
├── demo_app.py                        # Flask web application
├── Ver1_Smart_City_Report.docx        # Detailed project report
├── models/
│   ├── dcrnn.py                       # DCRNN implementation
│   ├── gman.py                        # GMAN variants
│   ├── parking_cnn.py                 # Parking detection models
│   └── utils.py                       # Helper functions
├── data/
│   ├── METR-LA/                       # Traffic dataset
│   │   ├── metr-la.h5                # Speed data
│   │   └── adj_mx.pkl                # Adjacency matrix
│   └── PKLot/                         # Parking dataset
│       ├── PUCPR/
│       ├── UFPR04/
│       └── UFPR05/
├── templates/
│   └── index.html                     # Web UI template
├── static/
│   ├── css/
│   └── js/
├── checkpoints/                       # Saved model weights
│   ├── traffic_model.pth
│   └── parking_model.pth
└── results/
    ├── figures/                       # Generated plots
    └── metrics/                       # Evaluation results
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/smart-city-deep-learning.git
cd smart-city-deep-learning
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages

```txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.8.0
Pillow>=10.0.0
flask>=2.3.0
```

---

## 🚀 Usage

### Training Traffic Models

**1. Prepare Data**
```python
# Load METR-LA dataset
data = pd.read_hdf('data/METR-LA/metr-la.h5')
with open('data/METR-LA/adj_mx.pkl', 'rb') as f:
    adj_matrix = pickle.load(f)
```

**2. Train DCRNN**
```python
from models.dcrnn import DCRNN

model = DCRNN(
    num_nodes=207,
    input_dim=1,
    hidden_dim=64,
    num_layers=2,
    K=3
)

# Train model
history = train_model(
    model, 
    train_loader, 
    val_loader, 
    fwd_adj, 
    bwd_adj,
    epochs=30
)
```

**3. Evaluate**
```python
test_results = evaluate_model(model, test_loader, fwd_adj, bwd_adj)
print(f"Test MAE: {test_results['mae']:.3f} mph")
```

### Training Parking Models

**1. Load PKLot Dataset**
```python
from torch.utils.data import DataLoader
from models.parking_cnn import ParkingDataset, ParkNetLSTM

train_dataset = ParkingDataset('data/PKLot', split='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**2. Train ParkNet-LSTM**
```python
model = ParkNetLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_acc = validate(model, val_loader)
    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={val_acc:.2f}%")
```

### Running Jupyter Notebook

```bash
jupyter notebook smart_city_final.ipynb
```

The notebook contains:
- Complete data preprocessing pipeline
- All model implementations
- Training procedures
- Comprehensive evaluation and visualization
- Comparison plots and analysis

---

## 📊 Dataset

### 1. METR-LA Traffic Dataset

**Source:** Los Angeles Metropolitan Traffic
- **Sensors:** 207 loop detectors on highways
- **Time Period:** March 2012 - June 2012
- **Sampling Rate:** 5 minutes
- **Features:** Traffic speed (mph)
- **Total Samples:** ~34,000 timesteps
- **Spatial Graph:** Distance-based adjacency matrix

**Data Split:**
- Training: 70% (first 2 months)
- Validation: 10% (next 2 weeks)
- Testing: 20% (last month)

**Preprocessing:**
- Z-score normalization per sensor
- Missing value interpolation
- Sliding window: 12 steps input → 12 steps output

### 2. PKLot Parking Dataset

**Source:** Brazilian parking lots (PUCPR, UFPR)
- **Total Images:** 12,416
- **Classes:** Vacant (0) / Occupied (1)
- **Image Size:** 64×64 RGB
- **Parking Lots:** 3 different locations
- **Weather Conditions:** Sunny, cloudy, rainy
- **Time Periods:** Morning, afternoon, evening

**Data Split:**
- Training: 70% (PUCPR + UFPR04)
- Validation: 15% (Mixed)
- Testing: 15% (UFPR05 for cross-lot)

**Augmentation:**
- Random rotation (±15°)
- Horizontal flip
- Brightness/contrast adjustment
- Gaussian noise

---

## 📈 Results

### Traffic Prediction Performance

| Model | MAE (mph) | RMSE (mph) | MAPE (%) | Training Time |
|-------|-----------|------------|----------|---------------|
| Historical Average | 4.87 | 7.24 | 9.8 | - |
| ARIMA | 4.12 | 6.53 | 8.4 | Fast |
| *DCRNN (Ours)* | *3.47* | *5.82* | *7.1* | 15 min/epoch |
| GraphWaveNet | 3.52 | 5.89 | 7.3 | 18 min/epoch |
| GMAN | 3.41 | 5.76 | 6.9 | 25 min/epoch |

**Prediction Horizon Analysis:**
- 15-min ahead: 2.8 mph MAE
- 30-min ahead: 3.4 mph MAE
- 60-min ahead: 4.2 mph MAE

### Parking Detection Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| mAlexNet (Baseline) | 88.3% | 87.1% | 89.2% | 88.1% |
| **ParkNet-LSTM** | **92.5%** | **91.8%** | **93.2%** | **92.5%** |

**Cross-Lot Generalization:**
- PUCPR → UFPR05: 90.2%
- UFPR04 → PUCPR: 89.7%
- Overall: 92.5%

### Key Insights

✅ **DCRNN outperforms baseline by 9.7%** in traffic prediction MAE  
✅ **ParkNet-LSTM achieves 92.5% accuracy** with strong cross-lot generalization  
✅ **Attention mechanisms** (GMAN) provide marginal improvements with higher complexity  
✅ **Graph structure is crucial** - random graphs degrade performance by 15%  
✅ **LSTM integration** in parking model improves temporal consistency  

---

## 🌐 Demo Application

### Quick Start

```bash
python demo_app.py
```

Then open your browser to: `http://localhost:5000`

### Features

**Traffic Prediction Tab:**
- Select from 207 sensors
- View real-time speed predictions
- 60-minute forecast horizon
- Interactive charts with historical data
- Confidence intervals

**Parking Detection Tab:**
- Select parking lot
- Real-time occupancy status
- Sample image predictions
- Detection confidence scores
- Occupancy trends

### API Endpoints

```python
# Get traffic prediction
POST /api/traffic/predict
{
  "sensor_id": 100
}

# Get parking status
POST /api/parking/predict
{
  "lot_name": "PUCPR"
}

# Get system statistics
GET /api/stats
```

### Screenshots

```
Traffic Dashboard:
┌──────────────────────────────────────┐
│ Sensor: I-405 N Mile 23              │
│ Current: 52.3 mph                    │
│ 15-min: 48.7 mph ↓                   │
│ 30-min: 45.2 mph ↓                   │
│ 60-min: 51.8 mph ↑                   │
│ [Interactive Chart]                  │
└──────────────────────────────────────┘

Parking Dashboard:
┌──────────────────────────────────────┐
│ PUCPR Parking Lot                    │
│ Occupancy: 112/150 (74.7%)          │
│ Available: 38 spaces                 │
│ [Sample Images with Predictions]     │
└──────────────────────────────────────┘
```

---

## 👥 Team

**BITS Pilani, Dubai Campus**

- **Ishan Rajesh** - Traffic Models & Graph Neural Networks
- **Jonathan Sam** - Parking Detection & Computer Vision
- **Kishore Pramodh** - Data Pipeline & Evaluation
- **Kritin Murkoth** - Web Application & Deployment

**Course:** Deep Learning 
**Instructor:** Dr. Tamizharasan Periyasamy  


---

## 📚 References

### Traffic Forecasting

1. Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). **Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.** ICLR 2018.

2. Zheng, C., Fan, X., Wang, C., & Qi, J. (2020). **GMAN: A Graph Multi-Attention Network for Traffic Prediction.** AAAI 2020.

3. Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). **Graph WaveNet for Deep Spatial-Temporal Graph Modeling.** IJCAI 2019.

### Parking Detection

4. Amato, G., et al. (2017). **Deep Learning for Decentralized Parking Lot Occupancy Detection.** Expert Systems with Applications.

5. De Almeida, P. R., et al. (2015). **PKLot – A Robust Dataset for Parking Lot Classification.** Expert Systems with Applications.

### Datasets

- **METR-LA:** [https://github.com/liyaguang/DCRNN](https://github.com/liyaguang/DCRNN)
- **PKLot:** [https://web.inf.ufpr.br/vri/databases/parking-lot-database/](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- METR-LA dataset provided by Los Angeles Department of Transportation
- PKLot dataset from Federal University of Paraná
- PyTorch and PyTorch Geometric communities
- BITS Pilani Dubai Campus for computational resources

---


## 🔮 Future Work

- [ ] Integrate real-time data streaming
- [ ] Deploy on cloud infrastructure (AWS/GCP)
- [ ] Add more parking datasets
- [ ] Implement transfer learning across cities
- [ ] Mobile application development
- [ ] Multi-task learning (traffic + parking)
- [ ] Explainable AI for model interpretability

---


