"""Lag-series features (sweep 5, R5 and L1).

Twelve monthly lags of fuel litres, fuel rands, non-fuel rands and basket
counts, plus the last thirteen weekly fuel totals. Lags that reach back before
the customer's first transaction are NaN rather than 0, so the model can tell
"no history" apart from "history with no spend" -- the distinction matters here
because tenure varies from weeks to the full 20 months.

Everything reads only rows strictly before the cutoff.
"""
import numpy as np, pandas as pd
import features as F

N_MONTH_LAGS = 12
N_WEEK_LAGS = 13


def build(d, cutoff, ids):
    t = pd.Timestamp(cutoff)
    h = d[d._time < t]
    f = pd.DataFrame(index=ids)
    first = h.groupby('ID')._time.min().reindex(ids)

    # monthly lags: lag 1 is the complete calendar month before the cutoff
    for k in range(1, N_MONTH_LAGS + 1):
        a = t - pd.DateOffset(months=k)
        z = t - pd.DateOffset(months=k - 1)
        w = h[(h._time >= a) & (h._time < z)]
        fu, nf = w[w.is_fuel], w[w.is_nonfuel]
        cols = {
            f'lag{k}_fuel_l': fu.groupby('ID').qty.sum(),
            f'lag{k}_fuel_r': fu.groupby('ID').amt.sum(),
            f'lag{k}_nf_r': nf.groupby('ID').amt.sum(),
            f'lag{k}_baskets': w.groupby('ID').TransactionId.nunique(),
        }
        # NaN where the window predates the customer's first transaction
        before_first = first >= z
        for name, s in cols.items():
            v = s.reindex(ids).fillna(0.0)
            f[name] = v.where(~before_first, np.nan)

    # weekly fuel totals, most recent first
    for k in range(1, N_WEEK_LAGS + 1):
        a = t - pd.Timedelta(days=7 * k)
        z = t - pd.Timedelta(days=7 * (k - 1))
        w = h[(h._time >= a) & (h._time < z) & h.is_fuel]
        v = w.groupby('ID').qty.sum().reindex(ids).fillna(0.0)
        f[f'wk{k}_fuel_l'] = v.where(~(first >= z), np.nan)

    # how many of the twelve monthly lags are observed at all
    f['lags_observed'] = f[[f'lag{k}_baskets' for k in range(1, N_MONTH_LAGS + 1)]] \
        .notna().sum(axis=1)
    f.columns = F._clean(f.columns)
    return f.replace([np.inf, -np.inf], np.nan)
