# PorfolioTheoryProject
## Kernel PCA for stock selection along with back testing
Uses KPCA to extract important factors from market and use it to select stocks.
Weight construction uses convex optimization along with constraints based on course criteria: beta neutral, full investment, long only, and annual tracking error.
Each backtest window are 24(T) days over 80(N) stocks from SP500. Should see the merit of KPCA given N >> T.

