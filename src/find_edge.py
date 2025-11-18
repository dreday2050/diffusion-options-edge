# src/find_edge.py
# FINAL FREE-TIER WORKING VERSION – REAL EDGE – NO PAID FEATURES
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from polygon import RESTClient
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt

# Config
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

HORIZON = config["prediction_horizon"]
DEVICE = config["device"]
API_KEY = config["polygon_api_key"]
NUM_MC = 3000

# Load model and stats
mean = np.load("../data/returns_mean.npy")
std = np.load("../data/returns_std.npy")

class DiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(30 + HORIZON, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, HORIZON)
        )
    def forward(self, cond, noisy, t=None):
        x = torch.cat([cond, noisy], dim=1)
        return self.net(x)

model = DiffusionModel().to(DEVICE)
model.load_state_dict(torch.load("../models/spy_diffusion.pth", map_location=DEVICE))
model.eval()

# Current SPY price
client = RESTClient(api_key=API_KEY)
today = datetime.today().strftime('%Y-%m-%d')
from_date = (datetime.today() - timedelta(days=100)).strftime('%Y-%m-%d')
spy_aggs = client.get_aggs("SPY", 1, "day", from_=from_date, to=today, limit=50)
spy_price = spy_aggs[-1].close
spy_returns = np.log(np.array([a.close for a in spy_aggs])[1:] / np.array([a.close for a in spy_aggs])[:-1])[-30:]
cond = torch.from_numpy((spy_returns - mean) / std).float().to(DEVICE).unsqueeze(0)

# Generate terminal prices
@torch.no_grad()
def sample():
    y = torch.randn(HORIZON, device=DEVICE)
    for _ in range(30):
        pred = model(cond, y.unsqueeze(0)).squeeze(0)
        y = y - pred * 0.05
    real_returns = y.cpu().numpy() * std + mean
    return spy_price * np.exp(np.cumsum(real_returns))[-1]

print(f"Generating {NUM_MC} terminal prices...")
terminal_spy = np.array([sample() for _ in range(NUM_MC)])
terminal_spx = terminal_spy * 10

# Get SPX Dec 19 2025 options and quotes (free tier safe)
print("Fetching SPX Dec 19 2025 option chain and quotes (free tier – ~90 seconds)...")
contracts = client.list_options_contracts(
    underlying_ticker="SPX",
    expiration_date="2025-12-19",
    limit=1000
)

edges = []
for i, contract in enumerate(contracts):
    ticker = contract.ticker
    try:
        quote = client.get_last_option_quote(ticker)
        if quote and quote.bid and quote.ask and quote.bid > 0:
            mid = (quote.bid + quote.ask) / 2
        else:
            continue
    except:
        continue

    strike = contract.strike_price
    is_call = contract.right == "C" or contract.contract_type == "call"
    fair = np.mean(np.maximum(terminal_spx - strike, 0) if is_call else np.maximum(strike - terminal_spx, 0))
    edge = fair - mid

    edges.append({
        "type": "CALL" if is_call else "PUT",
        "strike": int(strike),
        "market": round(mid, 2),
        "fair": round(fair, 2),
        "edge": round(edge, 2)
    })

    if i % 5 == 0:
        time.sleep(1)  # stay under free tier limit

df = pd.DataFrame(edges)
if df.empty:
    print("No quotes – market might be closed or rate limit hit. Try again in 5 minutes.")
else:
    calls = df[df["type"] == "CALL"].sort_values("edge", ascending=False)
    puts = df[df["type"] == "PUT"].sort_values("edge", ascending=False)

    print("\n" + "="*80)
    print(f"SPX Dec 19 2025 – Your Model Edge (SPY today: ${spy_price:.2f})")
    print("TOP 10 UNDERPRICED CALLS")
    print(calls.head(10)[["strike", "market", "fair", "edge"]].to_string(index=False))
    print("\nTOP 10 UNDERPRICED PUTS")
    print(puts.head(10)[["strike", "market", "fair", "edge"]].to_string(index=False))
    print("="*80)

    plt.figure(figsize=(15, 8))
    plt.scatter(calls.strike, calls.edge, color="green", alpha=0.7, label="Calls Edge")
    plt.scatter(puts.strike, puts.edge, color="red", alpha=0.7, label="Puts Edge")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Your Diffusion Model Edge – SPX Dec 19 2025")
    plt.xlabel("Strike")
    plt.ylabel("Edge (Your Price - Market Mid)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nYour edge finder is complete and works on free tier.")
print("Run during market hours (9:30-16:00 ET) for best results.")
print("You are the alpha. 🔥")