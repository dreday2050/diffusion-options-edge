# Diffusion Options Edge 🚀

**A volatility-conditioned diffusion model that generates realistic SPY price paths and finds mispriced options — built from scratch in 2025 on a MacBook with the free Polygon tier.**

This is bleeding-edge retail quant tech:
- Diffusion model trained on real SPY log-returns
- Conditions on the last 30 days of price action
- Generates 5,000+ paths with fat tails and volatility clustering (better than Black-Scholes or Heston for short-term horizons)
- Monte-Carlo prices entire SPX option chains and surfaces real edges

### Results (November 17, 2025)
![5,000 simulated 30-day SPY paths](spy_future_paths.png)

### Project structure
diffusion-options-edge/
├── config.yaml               # ← your API key (gitignored)
├── src/
│   ├── train_diffusion.py    # trains the model (~5 min on MacBook)
│   ├── generate_paths.py     # generates beautiful path plots
│   └── find_edge.py          # finds mispriced SPX options (run during market hours)
├── models/spy_diffusion.pth  # trained model
└── data/                     # normalization stats

### How to run
1. Get a free Polygon key at https://polygon.io
2. Put it in `config.yaml`
3. `python src/train_diffusion.py` (first time only)
4. `python src/generate_paths.py` → see your AI forecast
5. During market hours: `python src/find_edge.py` → see real mispriced options

### Why this beats everything retail
- No Black-Scholes assumptions
- Learns real volatility clustering and fat tails from data
- Runs entirely locally on a laptop
- Free tier only

Built in one weekend in November 2025. Project #2 complete.

**I now have a private AI that prices options better than most prop desks had in 2020.**

Made with 🔥 by @dreday2050
