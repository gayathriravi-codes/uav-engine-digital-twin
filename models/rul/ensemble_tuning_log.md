RUL Ensemble Tuning Log

Baseline v3 - committed before tuning pass
- Variants: 5, hidden_size 16/24/32/40/48, dropout 0.0-0.25, seeds 0-4, 0-2 dropped features
- Ensemble RMSE mean of variants: 40.73
- Per-variant RMSE: 49.81, 35.25, 35.89, 45.15, 37.52
- Sanity check std 5 test windows: approx 23-25 min
- Sanity check point_est / lower_bound: approx 197 / 152
- Known issues: variant 1 hidden_size=16 undersized; variant 4 dropped egt+oil_pressure together
- Split: train/test only, no validation set, tuning happened on test set so far - being fixed this pass
