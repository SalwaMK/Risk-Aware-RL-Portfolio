# Risk Aware PPO Portfolio Optimization

Reformulating the RL reward function for risk-aware portfolio optimization using PPO.

## Reward Function

$$r_t = R_t \;-\; \lambda \cdot R_t^{2} \;-\; \beta \cdot \max(0,\,-R_t)^{2}$$

- **λ** — penalises return variance (both directions)
- **β** — asymmetric downside penalty (losses hurt more than gains)

---

## Notebooks

| Notebook | Description |
|---|---|
| `ppo_portfolio_baseline.ipynb` | Baseline — 200k timesteps, 3 risk configs, single seed |
| `ppo_portfolio_enhanced.ipynb` | Enhanced — 1M timesteps, 4×4 grid search, 5 seeds, 80 total runs |

---

## Dataset

8 US stocks (AAPL, MSFT, GOOGL, JPM, XOM, JNJ, AMZN, TSLA) — 2013 to 2023  
Train: 2013–2020 | Test: 2021–2023

---

## Version Comparison

| Feature | Baseline | Enhanced |
|---|---|---|
| Timesteps | 200,000 | 1,000,000 |
| Seeds | 1 | 5 |
| Configurations | 3 (λ=β coupled) | 16 (4×4 independent grid) |
| Training episodes | ~30 (undertrained) | ~500 (fully converged) |
| Network size | 64×64 MLP | 256×256 MLP |
| Observation window | 20 days | 60 days |
| Baselines | Equal-weight only | Equal-weight + SPY + Min-Variance |
| Metrics | Sharpe, Drawdown | + Sortino, Calmar, Bootstrap CI |
| Transaction costs | No | Yes (0.1% per rebalance) |
| Market regime analysis | No | Bull (2021) / Bear (2022) breakdown |
| Statistical validation | No | Mean ± std + 95% bootstrap CI |

---

## Results

### Version 1 — Baseline (single seed, 200k timesteps)

| Strategy | Sharpe | Ann. Return | Max Drawdown |
|---|---|---|---|
| Equal-Weight | 0.836 | 16.96% | -27.16% |
| PPO No-Risk (λ=0, β=0) | 0.612 | 11.45% | -34.76% |
| PPO Moderate (λ=0.1, β=0.1) | 0.848 | 17.22% | -24.69% |
| PPO High-Risk (λ=0.5, β=0.5) | **1.021** | **22.81%** | -25.31% |

> Single seed — results are not statistically validated. Agents were undertrained (~30 episodes).

---

### Version 2 — Enhanced (5 seeds, 1M timesteps, full grid search)

**Key finding:** λ (variance penalty) improves risk-adjusted returns. β (downside penalty) becomes counterproductive beyond 0.3.

#### Best configurations from 4×4 grid (mean Sharpe across 5 seeds):

| Configuration | Mean Sharpe | vs No-Risk Baseline |
|---|---|---|
| λ=0.5, β=0 | **0.6748** | +12.0% |
| λ=0, β=0.3 | 0.6470 | +7.4% |
| λ=0.1, β=0.3 | 0.6562 | +9.0% |
| λ=0, β=0 (baseline) | 0.6023 | — |
| λ=0.1, β=0.5 | 0.4909 | -18.5% |

#### Representative agents (mean ± std across 5 seeds):

| Strategy | Sharpe | Ann. Return | Max Drawdown |
|---|---|---|---|
| Equal-Weight | 0.836 | 16.96% | -27.16% |
| PPO No-Risk (λ=0, β=0) | 0.60 ± std | ~13% | ~-20% |
| PPO Moderate (λ=0.1, β=0.1) | 0.58 ± std | ~13% | ~-20% |
| PPO High-Risk (λ=0.5, β=0.5) | 0.63 ± std | ~13% | ~-20% |

> Full metrics table with mean ± std available in notebook output.

---

## What Improved

- **Training quality:** Agents now complete ~500 episodes vs ~30 fully converged policies
- **Statistical reliability:** All results validated across 5 seeds with bootstrap confidence intervals
- **Key scientific finding:** λ and β have opposing effects, variance penalization helps, excessive downside penalization hurts
- **Heatmap spread:** Sharpe range widened from 0.009 (meaningless) to 0.18 (meaningful pattern)
- **Reproducibility:** Full seed control, model checkpointing, resume support

---

## Setup

```bash
pip install stable-baselines3 gymnasium yfinance pandas numpy matplotlib seaborn torch
```

---

## License

MIT
