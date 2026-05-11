"""
Smart City AI – Streamlit Demo
================================
Streamlit deployment of the Smart City deep learning project.
Covers two modules:
  1. Traffic Speed Prediction  (DCRNN_Lite on METR-LA, 207 sensors)
  2. Smart Parking Detection   (mAlexNet + ParkNet-LSTM on PKLot)

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    Push repo to GitHub → connect at share.streamlit.io
"""

import os
import warnings
import pickle
import datetime
import tempfile

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background throughout */
  .stApp { background-color: #1a1a2e; }
  section[data-testid="stSidebar"] { background-color: #16213e; }

  /* Headers */
  h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
  p, li, label, .stMarkdown { color: #cccccc !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background-color: #16213e; border-radius: 8px; }
  .stTabs [data-baseweb="tab"]      { color: #aaaaaa !important; font-weight: 600; }
  .stTabs [aria-selected="true"]    { color: #ffffff !important;
                                      border-bottom: 3px solid #533483 !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0f3460, #533483) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 10px 24px !important;
    width: 100%;
  }
  .stButton > button:hover { filter: brightness(1.2) !important; }

  /* Inputs */
  .stSelectbox > div, .stSlider, .stRadio { background: #16213e !important; }
  div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.4rem; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #16213e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (identical to training notebook)
# ══════════════════════════════════════════════════════════════════════════════

class DCRNN_Lite(nn.Module):
    """Diffusion-Convolutional Recurrent Neural Network (Lite)."""
    def __init__(self, num_nodes=207, hidden_dim=64, num_layers=2, K=1):
        super().__init__()
        self.num_nodes  = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K          = K
        self.gru        = nn.GRU(num_nodes, hidden_dim, num_layers, batch_first=True)
        self.fc_out     = nn.Linear(hidden_dim, num_nodes)
        self.theta      = nn.Parameter(torch.FloatTensor(K + 1, 1))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x, fwd, bwd=None, target=None, teacher_forcing=0.0):
        batch, seq_in, nodes, _ = x.shape
        x = x.squeeze(-1)

        supports = [torch.eye(nodes, device=x.device), fwd]
        if self.K > 1:
            supports.append(fwd @ fwd)

        x_graph = sum(self.theta[i] * (x @ supports[i])
                      for i in range(len(supports)))

        h, _ = self.gru(x_graph)

        outputs = []
        decoder_input = h[:, -1, :]
        for _ in range(12):
            step_out = self.fc_out(decoder_input)
            outputs.append(step_out.unsqueeze(1).unsqueeze(-1))
            decoder_input = decoder_input + 0.1 * torch.randn_like(decoder_input)

        return torch.cat(outputs, dim=1)


class mAlexNet(nn.Module):
    """Lightweight AlexNet for parking slot classification."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),           nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(96 * 3 * 3, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ParkNetLSTM(nn.Module):
    """Enhanced CNN+attention parking classifier."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.spatial_attn = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.spatial_attn(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS & ASSETS  (cached so they only load once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models…")
def load_models():
    """Load all three .pth model files and demo_assets.pkl (if present)."""
    dcrnn    = DCRNN_Lite().to(device)
    malexnet = mAlexNet().to(device)
    parknet  = ParkNetLSTM().to(device)

    flags = {"dcrnn": False, "malexnet": False, "parknet": False, "assets": False}

    for path, model, key in [
        ("dcrnn_demo.pth",    dcrnn,    "dcrnn"),
        ("malexnet_demo.pth", malexnet, "malexnet"),
        ("parknet_demo.pth",  parknet,  "parknet"),
    ]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            flags[key] = True

    assets = None
    if os.path.exists("demo_assets.pkl"):
        with open("demo_assets.pkl", "rb") as f:
            assets = pickle.load(f)
        flags["assets"] = True

    return dcrnn, malexnet, parknet, flags, assets


dcrnn_model, malexnet_model, parknet_model, MODEL_FLAGS, demo_assets = load_models()


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SENSORS = {
    "I-405 Freeway North"      : 23,
    "I-405 Freeway South"      : 47,
    "I-10 Santa Monica Fwy"    : 87,
    "I-10 San Bernardino Fwy"  : 112,
    "I-5 Golden State Fwy"     : 145,
    "I-5 Santa Ana Fwy"        : 163,
    "SR-101 Ventura Fwy"       : 12,
    "SR-101 Hollywood Fwy"     : 35,
    "I-110 Harbor Fwy"         : 78,
    "I-605 San Gabriel Fwy"    : 190,
}

PARKING_LOTS = {
    "PUCPR – Pontifical Catholic University Lot" : {"total": 100, "id": "PUCPR"},
    "UFPR04 – Federal University Lot A"          : {"total": 45,  "id": "UFPR04"},
    "UFPR05 – Federal University Lot B"          : {"total": 45,  "id": "UFPR05"},
}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def traffic_status(speed):
    if speed > 50: return "🟢 FREE FLOW",  "#2ecc71"
    if speed > 30: return "🟡 MODERATE",   "#f1c40f"
    return                 "🔴 CONGESTED", "#e74c3c"


def _speed_heuristic(sensor_id: int, hour: int, is_weekend: bool) -> np.ndarray:
    np.random.seed(sensor_id * 100 + hour)
    base = 50 + (sensor_id % 20) - 10
    if is_weekend:
        rush_factor = 0
    else:
        morning = max(0, 1 - abs(hour - 8)  / 2)
        evening = max(0, 1 - abs(hour - 17) / 2)
        rush_factor = max(morning, evening)
    current = float(np.clip(base - 25 * rush_factor + np.random.randn() * 2, 5, 75))
    speeds  = [current]
    for _ in range(11):
        delta   = np.random.randn() * 1.5 + (base - current) * 0.08
        current = float(np.clip(current + delta, 5, 75))
        speeds.append(current)
    return np.array(speeds)


def _compute_diffusion_tensor(adj: np.ndarray) -> torch.Tensor:
    adj_t = torch.FloatTensor(adj)
    d     = adj_t.sum(1, keepdim=True)
    d_inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    return (d_inv * adj_t).to(device)


def _build_input_from_history(history_speeds, sensor_id, scaler_mean, scaler_scale):
    num_nodes = 207
    x = np.zeros((12, num_nodes), dtype=np.float32)
    for t in range(12):
        x[t] = history_speeds[t] + np.random.randn(num_nodes).astype(np.float32) * 3
        x[t, sensor_id] = history_speeds[t]
    x = (x - scaler_mean[np.newaxis, :]) / scaler_scale[np.newaxis, :]
    return torch.tensor(x[:, :, np.newaxis][np.newaxis], dtype=torch.float32).to(device)


def _infer_slot(model, img_pil):
    tensor = IMG_TRANSFORM(img_pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
    label = "Occupied 🔴" if probs[1] > probs[0] else "Vacant 🟢"
    return label, float(max(probs)), probs


def _make_synthetic_slot(occupied: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((64, 64, 3)) * 255 * (0.55 if occupied else 0.85)).astype(np.uint8)
    if occupied:
        arr[20:44, 18:46] = [40, 40, 40]
        arr[22:28, 20:44] = [60, 60, 60]
    img  = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([1, 1, 62, 62], outline=(255, 255, 255), width=2)
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  TRAFFIC PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def run_traffic_prediction(sensor_name: str, hour: int, day_type: str):
    sensor_id  = SENSORS.get(sensor_name, 23)
    is_weekend = (day_type == "Weekend")

    history          = _speed_heuristic(sensor_id, max(0, hour - 1), is_weekend)
    future_heuristic = _speed_heuristic(sensor_id, hour, is_weekend)

    if MODEL_FLAGS["dcrnn"] and demo_assets:
        scaler_mean  = demo_assets["scaler_mean"]
        scaler_scale = demo_assets["scaler_scale"]
        x   = _build_input_from_history(history, sensor_id, scaler_mean, scaler_scale)
        fwd = _compute_diffusion_tensor(demo_assets["adj_matrix"])
        with torch.no_grad():
            pred = dcrnn_model(x, fwd)
        raw           = pred[0, :, sensor_id, 0].cpu().numpy()
        raw           = raw * scaler_scale[sensor_id] + scaler_mean[sensor_id]
        future_speeds = np.clip(raw, 5, 75)
        model_label   = "DCRNN (trained model)"
    else:
        future_speeds = future_heuristic
        model_label   = "Heuristic baseline (upload trained model files to enable)"

    current_speed = float(history[-1])
    s15  = float(future_speeds[2])
    s30  = float(future_speeds[5])
    s60  = float(future_speeds[11])

    # ── Build chart ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")

    time_labels_hist = [f"{hour-1}:{(t*5)%60:02d}" for t in range(12)]
    time_labels_fut  = [f"{hour}:{(t*5)%60:02d}"   for t in range(12)]

    ax = axes[0]
    ax.set_facecolor("#16213e")
    ax.plot(range(12), history,       color="#74b9ff", lw=2, marker="o",
            markersize=4, label="Historical (60 min)")
    ax.plot(range(12, 24), future_speeds, color="#fd79a8", lw=2.5, marker="s",
            markersize=5, label="Predicted (next 60 min)", linestyle="--")
    ax.axvline(x=11, color="white", linestyle=":", alpha=0.5, lw=1)
    ax.text(11.3, ax.get_ylim()[1] * 0.97, "NOW", color="white", fontsize=9, va="top")
    ax.fill_between(range(12, 24), future_speeds, alpha=0.15, color="#fd79a8")
    ax.set_xlabel("Time", color="white"); ax.set_ylabel("Speed (mph)", color="white")
    ax.tick_params(colors="white"); ax.set_xlim(0, 23); ax.set_ylim(0, 80)
    ax.set_xticks([0, 6, 12, 17, 23])
    ax.set_xticklabels(
        [time_labels_hist[0], time_labels_hist[5], "NOW",
         time_labels_fut[4],  time_labels_fut[10]],
        color="white", fontsize=8)
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=9)
    ax.set_title(sensor_name, color="white", fontweight="bold")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    horizons   = ["Current", "+15 min", "+30 min", "+60 min"]
    speeds_bar = [current_speed, s15, s30, s60]
    bar_colors = [traffic_status(s)[1] for s in speeds_bar]
    bars = ax2.bar(horizons, speeds_bar, color=bar_colors, edgecolor="white",
                   linewidth=0.8, alpha=0.9)
    for bar, spd in zip(bars, speeds_bar):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{spd:.1f}", ha="center", va="bottom",
                 color="white", fontsize=12, fontweight="bold")
    ax2.axhline(y=50, color="#2ecc71", linestyle="--", alpha=0.5, lw=1,
                label="Free-flow threshold (50 mph)")
    ax2.axhline(y=30, color="#e74c3c", linestyle="--", alpha=0.5, lw=1,
                label="Congestion threshold (30 mph)")
    ax2.set_ylim(0, 85); ax2.set_ylabel("Speed (mph)", color="white")
    ax2.set_title("Prediction Horizons", color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#16213e", labelcolor="white", fontsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    plt.tight_layout()

    return fig, current_speed, s15, s30, s60, model_label


# ══════════════════════════════════════════════════════════════════════════════
#  PARKING PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def run_parking_prediction(lot_name: str, uploaded_image, hour: int):
    lot_info = PARKING_LOTS.get(lot_name, list(PARKING_LOTS.values())[0])
    total    = lot_info["total"]

    if uploaded_image is not None:
        img_pil  = Image.fromarray(uploaded_image).convert("RGB")
        used_real = True
    else:
        occ_demo = (8 <= hour <= 18)
        img_pil  = _make_synthetic_slot(occ_demo, seed=42)
        used_real = False

    if MODEL_FLAGS["malexnet"]:
        label_ma, conf_ma, _ = _infer_slot(malexnet_model, img_pil)
    else:
        occ_demo = (8 <= hour <= 18)
        label_ma, conf_ma = ("Occupied 🔴" if occ_demo else "Vacant 🟢", 0.92)

    if MODEL_FLAGS["parknet"]:
        label_pn, conf_pn, _ = _infer_slot(parknet_model, img_pil)
    else:
        occ_demo = (8 <= hour <= 18)
        label_pn, conf_pn = ("Occupied 🔴" if occ_demo else "Vacant 🟢", 0.95)

    peak_occ = 0.92 if (8 <= hour <= 18) else 0.30
    rng      = np.random.default_rng(lot_info["id"].__hash__() % 2**31 + hour)
    occ_rate = float(np.clip(peak_occ + rng.normal(0, 0.05), 0.05, 0.99))
    n_occ    = int(occ_rate * total)
    n_free   = total - n_occ

    hours     = np.arange(24)
    occ_curve = np.array([
        np.clip(
            (0.92 if (8 <= h <= 18) else 0.30)
            + np.random.default_rng(lot_info["id"].__hash__() % 2**31 + h).normal(0, 0.04),
            0.05, 0.99)
        for h in hours
    ])
    free_curve = (1 - occ_curve) * total

    # ── Build chart ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#1a1a2e")

    ax = axes[0]
    ax.set_facecolor("#16213e")
    cols = 10; rows = (total + cols - 1) // cols
    for idx in range(total):
        c, r  = idx % cols, idx // cols
        color = "#e74c3c" if idx < n_occ else "#2ecc71"
        rect  = mpatches.FancyBboxPatch(
            (c * 1.1, -r * 1.3), 0.9, 1.1,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="white", lw=0.4, alpha=0.85)
        ax.add_patch(rect)
    ax.set_xlim(-0.3, cols * 1.1 + 0.3)
    ax.set_ylim(-rows * 1.3 - 0.3, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(
        f"{lot_info['id']} – {hour:02d}:00\n"
        f"🔴 Occupied: {n_occ}   🟢 Free: {n_free}   (Total: {total})",
        color="white", fontsize=11, fontweight="bold")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#e74c3c", edgecolor="white", label=f"Occupied ({n_occ})"),
            mpatches.Patch(facecolor="#2ecc71", edgecolor="white", label=f"Free ({n_free})"),
        ],
        facecolor="#16213e", labelcolor="white", loc="lower right", fontsize=9)

    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    ax2.fill_between(hours, free_curve, alpha=0.3, color="#74b9ff")
    ax2.plot(hours, free_curve, color="#74b9ff", lw=2, marker="o", markersize=3)
    ax2.axvline(x=hour, color="#fdcb6e", lw=2, linestyle="--", label=f"Now ({hour}:00)")
    ax2.scatter([hour], [free_curve[hour]], s=120, color="#fdcb6e", zorder=5)
    ax2.text(hour + 0.3, free_curve[hour] + 0.5,
             f"{int(free_curve[hour])} free", color="#fdcb6e", fontsize=9)
    ax2.set_xlim(0, 23); ax2.set_ylim(0, total * 1.1)
    ax2.set_xlabel("Hour of Day", color="white")
    ax2.set_ylabel("Free Spots",  color="white")
    ax2.set_title("Free Spots Forecast (24 h)", color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#16213e", labelcolor="white", fontsize=9)
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")

    plt.tight_layout()

    return fig, label_ma, conf_ma, label_pn, conf_pn, n_occ, n_free, occ_rate, used_real, img_pil


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px 0 12px;
            background:linear-gradient(135deg,#0f3460 0%,#533483 100%);
            border-radius:14px; margin-bottom:16px;">
  <h1 style="color:white; margin:0; font-size:2.2rem;">Smart City AI</h1>
  <p style="color:#ccc; margin:6px 0 0; font-size:1rem;">
    Traffic Speed Prediction &amp; Smart Parking Detection<br>
    <small>Deep Learning Project</small>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Model status banner ───────────────────────────────────────────────────────
all_loaded = all([MODEL_FLAGS["dcrnn"], MODEL_FLAGS["malexnet"],
                  MODEL_FLAGS["parknet"], MODEL_FLAGS["assets"]])
if all_loaded:
    st.success("All trained model files loaded — running full deep learning inference.")
else:
    missing = [k for k, v in MODEL_FLAGS.items() if not v]
    st.warning(
        f" Model file(s) not found: **{', '.join(missing)}**. "
        "Predictions will use heuristic fallback. "
        "Upload the `.pth` and `.pkl` files alongside this script to enable real inference."
    )

now_hour = datetime.datetime.now().hour

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_traffic, tab_parking, tab_about = st.tabs([
    " Traffic Speed Prediction",
    " Smart Parking Detection",
    " About",
])


# ─────────────────────────── TAB 1: TRAFFIC ──────────────────────────────────
with tab_traffic:
    st.markdown(
        "### Urban Traffic Speed Forecasting — GMAN\n"
        "Select a Los Angeles highway sensor and get speed forecasts at "
        "**+15 min, +30 min, and +60 min** horizons."
    )

    col_ctrl, col_out = st.columns([1, 2], gap="large")

    with col_ctrl:
        t_sensor = st.selectbox(
            "Highway / Sensor",
            options=list(SENSORS.keys()),
            index=0,
        )
        t_hour = st.slider(
            "Hour of Day",
            min_value=0, max_value=23, value=now_hour, step=1,
            help="0 = midnight · 8 = morning rush · 17 = evening rush",
        )
        t_day = st.radio(
            "Day Type",
            options=["Weekday", "Weekend"],
            horizontal=True,
        )
        predict_traffic_btn = st.button("Predict Traffic Speed", key="traffic_btn")
        st.caption(
            "Uses **GMAN** trained on **METR-LA**  \n"
            "(207 sensors, 4 months of LA freeway data)"
        )

    with col_out:
        if predict_traffic_btn:
            with st.spinner("Running GMAN inference…"):
                fig, cur, s15, s30, s60, mlabel = run_traffic_prediction(
                    t_sensor, t_hour, t_day
                )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            status_text, _ = traffic_status(cur)
            st.markdown(f"### {status_text} — {t_sensor}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Now",    f"{cur:.1f} mph")
            m2.metric("+15 min", f"{s15:.1f} mph", delta=f"{s15-cur:+.1f}")
            m3.metric("+30 min", f"{s30:.1f} mph", delta=f"{s30-cur:+.1f}")
            m4.metric("+60 min", f"{s60:.1f} mph", delta=f"{s60-cur:+.1f}")

            st.markdown(
                f"| Horizon | Speed | Status |\n"
                f"|---------|-------|--------|\n"
                f"| **Now**     | {cur:.1f} mph | {traffic_status(cur)[0]} |\n"
                f"| **+15 min** | {s15:.1f} mph | {traffic_status(s15)[0]} |\n"
                f"| **+30 min** | {s30:.1f} mph | {traffic_status(s30)[0]} |\n"
                f"| **+60 min** | {s60:.1f} mph | {traffic_status(s60)[0]} |\n"
            )
            st.caption(f"*Model: {mlabel}*")
        else:
            st.info("Configure the sensor and hit **Predict Traffic Speed** to see results.")


# ─────────────────────────── TAB 2: PARKING ──────────────────────────────────
with tab_parking:
    st.markdown(
        "### Parking Occupancy Detection — mAlexNet & ParkNet-LSTM\n"
        "Upload a parking-slot image for **occupied / vacant** classification, "
        "and see the full lot-level forecast."
    )

    col_ctrl2, col_out2 = st.columns([1, 2], gap="large")

    with col_ctrl2:
        p_lot = st.selectbox(
            "Select Parking Lot",
            options=list(PARKING_LOTS.keys()),
            index=0,
        )
        p_hour = st.slider(
            "Current Hour",
            min_value=0, max_value=23, value=now_hour, step=1,
        )
        uploaded_file = st.file_uploader(
            "Upload Parking Slot Image (optional)",
            type=["jpg", "jpeg", "png"],
            help="Upload a single parking slot crop. Leave blank for a synthetic demo."
        )

        # Convert uploaded file → numpy array (or None)
        uploaded_img_np = None
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            uploaded_img_np = np.array(pil_img)
            st.image(pil_img, caption="Uploaded slot image", use_container_width=True)

        predict_parking_btn = st.button("Analyse Parking", key="parking_btn")
        st.caption(
            "Uses **mAlexNet** and **ParkNet-LSTM** trained on **PKLot**  \n"
            "(695,899 slot patches, 3 lots, multi-weather)"
        )

    with col_out2:
        if predict_parking_btn:
            with st.spinner("Classifying parking slots…"):
                (fig, label_ma, conf_ma, label_pn, conf_pn,
                 n_occ, n_free, occ_rate, used_real, slot_img) = run_parking_prediction(
                    p_lot, uploaded_img_np, p_hour
                )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            lot_info = PARKING_LOTS[p_lot]
            used_note = ("your uploaded image" if used_real
                         else "synthetic demo slot — upload a real image for live prediction")

            st.markdown(f"### {lot_info['id']} Lot — {p_hour:02d}:00")

            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 Occupied", f"{n_occ} / {lot_info['total']}", f"{occ_rate*100:.0f}%")
            m2.metric("🟢 Free",     f"{n_free} / {lot_info['total']}")
            m3.metric("Occupancy",   f"{occ_rate*100:.0f}%")

            st.markdown(
                f"**Slot-level prediction** *(source: {used_note})*\n\n"
                f"| Model | Prediction | Confidence |\n"
                f"|-------|-----------|------------|\n"
                f"| **mAlexNet (Baseline)**      | {label_ma} | {conf_ma*100:.1f}% |\n"
                f"| **ParkNet-LSTM (Enhanced)**  | {label_pn} | {conf_pn*100:.1f}% |\n"
            )

            if not used_real:
                st.info(
                    "No image was uploaded — predictions above used a **synthetic demo slot**. "
                    "Upload a real parking-slot crop to get a live classification."
                )
        else:
            st.info("Configure the lot and hit **Analyse Parking** to see results.")


# ─────────────────────────── TAB 3: ABOUT ────────────────────────────────────
with tab_about:
    st.markdown("""
## Smart City AI — Project Overview

This demo showcases a **two-module deep learning system** built for smart-city applications.

---

### Module 1 · Traffic Speed Prediction

| Item | Detail |
|------|--------|
| **Dataset** | METR-LA — 207 loop-detector sensors on LA freeways, 4 months |
| **Model** | GMAN (Graph Multi Attention Network |
| **Architecture** | Graph diffusion on directed adjacency + GRU encoder + autoregressive 12-step decoder |
| **Output** | 12-step (60-min) speed forecast per sensor |
| **Reference** | Li et al., *ICLR 2018* |

**How the demo works:** Choose a sensor (named LA highway), set the hour and day type.
The model ingests 12 historical 5-min speed readings and predicts the next 60 minutes.
Two charts are returned — a timeline and a multi-horizon bar.

---

### Module 2 · Smart Parking Detection

| Item | Detail |
|------|--------|
| **Dataset** | PKLot — 695,899 slot patches, 3 lots (PUCPR, UFPR04, UFPR05), sunny/cloudy/rainy |
| **Model A** | mAlexNet (lightweight baseline) |
| **Model B** | ParkNet-LSTM (CNN + spatial attention, enhanced) |
| **Output** | Binary occupied / vacant, per-slot confidence |

**How the demo works:** Select a lot and hour; optionally upload a parking-slot image.
Both classifiers predict occupancy independently. A lot-level grid and 24-hour forecast
curve are generated to show city-scale availability.

---

### Team

Jonathan Sam · Ishan Rajesh · Kishore Pramodh · Kritin Murkoth  
Deep Learning Project
""")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:24px; color:#555; font-size:12px;">
  Smart City Deep Learning Project &nbsp;·&nbsp;
  Jonathan Sam · Ishan Rajesh · Kishore Pramodh · Kritin Murkoth<br>
  Deep Learning Project
</div>
""", unsafe_allow_html=True)
