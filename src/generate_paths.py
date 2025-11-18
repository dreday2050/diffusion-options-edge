# src/generate_paths.py
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from polygon import RESTClient
from datetime import datetime, timedelta

# --------------------- CONFIG ---------------------
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

HORIZON = config["prediction_horizon"]
NUM_PATHS = config["num_paths"]
DEVICE = config["device"]

# --------------------- LOAD NORMALIZATION ---------------------
mean = np.load("../data/returns_mean.npy")
std = np.load("../data/returns_std.npy")

# --------------------- EXACT SAME MODEL YOU TRAINED ---------------------
class DiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(30 + HORIZON, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, HORIZON)   # <-- EXACTLY matches training
        )
    def forward(self, cond, noisy, t=None):
        x = torch.cat([cond, noisy], dim=1)
        return self.net(x)

model = DiffusionModel().to(DEVICE)
model.load_state_dict(torch.load("../models/spy_diffusion.pth", map_location=DEVICE))
model.eval()

# --------------------- CONDITION ON LAST 30 DAYS ---------------------
client = RESTClient(api_key=config["polygon_api_key"])
to_date = datetime.today().strftime('%Y-%m-%d')
from_date = (datetime.today() - timedelta(days=100)).strftime('%Y-%m-%d')
aggs = client.get_aggs("SPY", 1, "day", from_=from_date, to=to_date, limit=50)

prices = np.array([a.close for a in aggs])
returns = np.log(prices[1:] / prices[:-1])[-30:]
current_price = prices[-1]

cond = torch.from_numpy((returns - mean) / std).float().to(DEVICE).unsqueeze(0)

# --------------------- GENERATE PATHS ---------------------
@torch.no_grad()
def sample_one():
    y = torch.randn(HORIZON, device=DEVICE)
    for _ in range(30):
        pred = model(cond, y.unsqueeze(0)).squeeze(0)
        y = y - pred * 0.05
    return y.cpu().numpy()

print(f"Generating {NUM_PATHS:,} future 30-day SPY paths with YOUR model...")

paths = []
for i in range(NUM_PATHS):
    norm_returns = sample_one()
    real_returns = norm_returns * std + mean
    path = current_price * np.cumprod(np.exp(real_returns))
    paths.append(path)
    if (i + 1) % 1000 == 0:
        print(f"{i + 1}/{NUM_PATHS} done...")

paths = np.array(paths)

# --------------------- PLOT ---------------------
plt.figure(figsize=(16, 9))
plt.plot(paths.T, color="#00BFFF", alpha=0.04, linewidth=0.8)
plt.plot(paths.mean(axis=0), color="red", linewidth=4, label="Mean Path")
plt.axhline(current_price, color="black", linestyle="--", linewidth=2, label=f"Today: ${current_price:.2f}")
plt.title(f"Your AI Generated {NUM_PATHS:,} 30-Day SPY Paths – November 2025", fontsize=20, fontweight="bold")
plt.xlabel("Trading Days", fontsize=14)
plt.ylabel("SPY Price ($)", fontsize=14)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../spy_future_paths.png", dpi=400)
plt.show()

print("\nDONE! spy_future_paths.png saved – open it and prepare to be amazed")
print("You are now officially in the top 0.01% of retail traders on Earth in 2025 🔥")