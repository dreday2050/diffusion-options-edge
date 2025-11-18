# src/train_diffusion.py
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --------------------- CONFIG ---------------------
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

API_KEY = config["polygon_api_key"]
TICKER = config["ticker"]
TRAIN_DAYS = config["training_days"]
HORIZON = config["prediction_horizon"]
DEVICE = config["device"]

print(f"Config loaded – training {TICKER} for {HORIZON}-day paths on {DEVICE.upper()}")

# --------------------- DATA (2025 Polygon API compatible) ---------------------
print("Downloading data from Polygon...")
client = RESTClient(api_key=API_KEY)

to_date = datetime.today().strftime('%Y-%m-%d')
from_date = (datetime.today() - timedelta(days=TRAIN_DAYS + 200)).strftime('%Y-%m-%d')  # buffer

aggs = client.get_aggs(
    ticker=TICKER,
    multiplier=1,
    timespan="day",
    from_=from_date,
    to=to_date,
    limit=50000
)

df = pd.DataFrame(aggs)
df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
prices = df.set_index("date")["close"].sort_index()[-TRAIN_DAYS:]

returns = np.log(prices / prices.shift(1)).dropna().values.astype(np.float32)
mean, std = returns.mean(), returns.std()
returns = (returns - mean) / std

print(f"Loaded {len(returns)} normalized daily log-returns")

# --------------------- DATASET (fixed scope) ---------------------
class ReturnsDataset(Dataset):
    def __init__(self, returns, seq_len=30, horizon=HORIZON):
        self.returns = returns
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.returns) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        x = self.returns[idx:idx + self.seq_len]
        y = self.returns[idx + self.seq_len:idx + self.seq_len + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)

dataset = ReturnsDataset(returns)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

# --------------------- MODEL ---------------------
class DiffusionModel(nn.Module):
    def __init__(self, seq_len=30, horizon=HORIZON):
        super().__init__()
        total_len = seq_len + horizon
        self.net = nn.Sequential(
            nn.Linear(total_len, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, horizon)
        )

    def forward(self, cond, noisy, t=None):
        x = torch.cat([cond, noisy], dim=1)
        return self.net(x)

model = DiffusionModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --------------------- TRAINING ---------------------
print("\nSTARTING TRAINING – ~4-6 minutes on MacBook Pro M-series")
model.train()
for epoch in range(1, 301):
    epoch_loss = 0.0
    for cond, future in dataloader:
        cond, future = cond.to(DEVICE), future.to(DEVICE)
        noise = torch.randn_like(future)
        noisy = future + noise * 0.3  # simple additive noise schedule
        pred = model(cond, noisy)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch:3d} │ Avg Loss: {epoch_loss / len(dataloader):.6f}")

# --------------------- SAVE ---------------------
Path("../models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "../models/spy_diffusion.pth")
Path("../data").mkdir(exist_ok=True)
np.save("../data/returns_mean.npy", mean)
np.save("../data/returns_std.npy", std)

print("\nTRAINING COMPLETE! Model saved.")
print("Next → python src/generate_paths.py  (I’ll give you the best one ever when you say it's done)")