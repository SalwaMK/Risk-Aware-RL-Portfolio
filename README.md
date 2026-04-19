# risk-aware-ppo-portfolio

Reformulating the RL reward function for risk-aware portfolio optimization using PPO.

## Reward Function

$$r_t = R_t \;-\; \lambda \cdot R_t^{2} \;-\; \beta \cdot \max(0,\,-R_t)^{2}$$

- **λ** — penalises return variance (both directions)
- **β** — asymmetric downside penalty (losses hurt more than gains)

## Notebooks

| Notebook | Description |
|---|---|
| `ppo_portfolio_baseline.ipynb` | Baseline — 200k timesteps, 3 risk configs, single seed |

## Dataset

8 US stocks (AAPL, MSFT, GOOGL, JPM, XOM, JNJ, AMZN, TSLA) — 2013 to 2023  
Train: 2013–2020 | Test: 2021–2023

## Results (single seed preview)
### PPO Portfolio Baseline

| Strategy | Sharpe | Ann. Return | Max Drawdown |
|---|---|---|---|
| Equal-Weight | 0.836 | 16.96% | -27.16% |
| PPO No-Risk | 0.612 | 11.45% | -34.76% |
| PPO Moderate | 0.848 | 17.22% | -24.69% |
| PPO High-Risk | **1.021** | **22.81%** | -25.31% |

## Setup

```bash
pip install stable-baselines3 gymnasium yfinance pandas numpy matplotlib torch
```

## License

MIT
