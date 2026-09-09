#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
base = "ml_swing_fullrun"
folders = [
    f"{base}/config",
    f"{base}/data/raw",
    f"{base}/data/reports",
    f"{base}/data/models",
    f"{base}/src",
    f"{base}/tools",
]
for d in folders:
    os.makedirs(d, exist_ok=True)
print("Folder siap:", folders)


# In[2]:


pip install --upgrade pandas


# In[3]:


import os, pandas as pd

log_path = "ml_swing_fullrun/data/reports/trade_log.csv"
cols = ['signal_date','entry_date','exit_date','ticker','ret','proba','rank']

def safe_read_trade_log(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        # tulis header kosong supaya bisa dibaca pandas
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return pd.read_csv(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # file berisi newline/whitespace saja → tulis ulang dgn header
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return pd.read_csv(path)

df_log = safe_read_trade_log(log_path)
print(f"{len(df_log)} baris terbaca dari trade_log.csv")
display(df_log.head())


# In[4]:


open("ml_swing_fullrun/requirements.txt","w").write(
"""pandas>=2.0
numpy>=1.26
scikit-learn>=1.3
xgboost>=2.0
pyyaml>=6.0
"""
)

open("ml_swing_fullrun/config/params.yaml","w").write(
"""universe:
  min_turnover_idr: 1500000000
  min_price: 100
  exclude_suspended: true
label:
  horizon_days: 5
  take_profit: 0.03
  stop_loss: 0.02
features:
  windows: [5,10,20,50,100,200]
  include_bandarmology: false
model:
  algo: xgboost
  n_estimators: 800
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.75
  reg_alpha: 0.001
  reg_lambda: 2.0
  scale_pos_weight: auto
cv:
  scheme: rolling
  lookback_months: 24
  test_months: 3
execution:
  costs_roundtrip_bps: 40
  slippage_bps: 10
report:
  top_n: 10
"""
)
print("requirements.txt & params.yaml dibuat.")


# In[5]:


# a) Perpanjang data jadi ~3 tahun (750 hari kerja)
from pathlib import Path
import pandas as pd, numpy as np
Path("ml_swing_fullrun/data/raw").mkdir(parents=True, exist_ok=True)

dates = pd.date_range("2022-01-03", periods=750, freq="B")
tickers = ["AALI","BBCA","TLKM","BMRI","ASII"]
rows=[]; rng=np.random.default_rng(321)
for t in tickers:
    base = rng.uniform(500, 2000)
    # tambah sedikit tren supaya breakout20 sering kejadian
    drift = np.linspace(0, 120, len(dates))
    price = base + drift + np.cumsum(rng.normal(0, 2.5, len(dates)))
    high = price + np.abs(rng.normal(2, 2, len(dates)))
    low  = price - np.abs(rng.normal(2, 2, len(dates)))
    openp= price + rng.normal(0,1.2, len(dates))
    vol  = rng.integers(2e5, 1e6, len(dates))
    val  = vol * price
    rows.append(pd.DataFrame({
        "date": dates, "ticker": t, "open": openp, "high": high, "low": low,
        "close": price, "volume": vol.astype(int), "value": val.astype(float), "adj_close": price
    }))
pd.concat(rows).to_csv("ml_swing_fullrun/data/raw/prices.csv", index=False)
print("Dummy data extended & trended.")

# b) Setting CV lebih ‘ringan’ dan top_n lebih besar
import yaml
p = yaml.safe_load(open("ml_swing_fullrun/config/params.yaml"))
p["cv"]["lookback_months"] = 12
p["cv"]["test_months"] = 2
p["report"]["top_n"] = 20
with open("ml_swing_fullrun/config/params.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(p, f, sort_keys=False)
print("params.yaml updated (lookback=12, test=2, top_n=20).")

# c) Rerun backtest
get_ipython().system('python ml_swing_fullrun/main.py backtest')


# In[6]:


import pandas as pd, os

def safe_read_csv(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            df = pd.read_csv(path)
            if df.shape[0] > 0:
                return df
            else:
                print(f"{path} ada header tapi tidak ada data.")
                return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"{path} kosong (EmptyDataError).")
            return pd.DataFrame()
    else:
        print(f"{path} tidak ada atau size 0 byte.")
        return pd.DataFrame()

bs = "ml_swing_fullrun/data/reports/backtest_summary.csv"
tl = "ml_swing_fullrun/data/reports/trade_log.csv"

df_bs = safe_read_csv(bs)
df_tl = safe_read_csv(tl)

display(df_bs.head())
display(df_tl.head())


# In[7]:


from pathlib import Path

fixed_main = r"""
import argparse, yaml
import pandas as pd
from pathlib import Path

from src.features import make_features
from src.label import make_labels_5d
from src.train import train_xgb
from src.score import score_proba
from src.split import rolling_time_splits
from src.backtest import simulate_trades, evaluate_trades

def load_prices(path='ml_swing_fullrun/data/raw/prices.csv'):
    df = pd.read_csv(path, parse_dates=['date'])
    required = ['date','ticker','open','high','low','close','volume','value','adj_close']
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f'Missing columns: {miss}')
    return df

def _build_next_day_map(prices):
    pr = prices.sort_values(['ticker','date']).copy()
    pr['next_date'] = pr.groupby('ticker')['date'].shift(-1)
    return pr.set_index(['date','ticker'])['next_date']

def run_backtest(params_path='ml_swing_fullrun/config/params.yaml'):
    params = yaml.safe_load(open(params_path, encoding='utf-8'))
    prices = load_prices()
    feat = make_features(prices)
    lbl = make_labels_5d(prices, tp=params['label']['take_profit'], sl=params['label']['stop_loss'], horizon=params['label']['horizon_days'])
    df = feat.merge(lbl, on=['date','ticker'], how='left').dropna(subset=['y'])

    results, all_trades = [], []
    pr_ohlc = prices.set_index(['date','ticker'])[['open','high','low','close']].sort_index()
    next_map = _build_next_day_map(prices)

    for tr_start, tr_end, te_start, te_end in rolling_time_splits(df, params['cv']['lookback_months'], params['cv']['test_months']):
        tr = df[(df['date']>=tr_start)&(df['date']<tr_end)].copy()
        te = df[(df['date']>=te_start)&(df['date']<te_end)].copy()

        y_tr = tr['y'].astype(int)
        drop_cols = ['y','open','high','low','close','adj_close','vwap']
        X_tr = tr.drop(columns=[c for c in drop_cols if c in tr.columns], errors='ignore').select_dtypes(include='number')
        X_te = te.drop(columns=[c for c in drop_cols if c in te.columns], errors='ignore').select_dtypes(include='number')

        model = train_xgb(X_tr, y_tr, params['model'])
        te = te.assign(proba = model.predict_proba(X_te)[:,1])
        te['rank'] = te.groupby('date')['proba'].rank(ascending=False, method='first')

        # hanya ambil sinyal yang punya next day (hindari sinyal di last day emiten)
        picks = te[te['rank']<=params['report']['top_n']][['date','ticker','proba','rank']].copy()
        picks['next_date'] = picks.set_index(['date','ticker']).index.map(next_map)
        picks = picks[picks['next_date'].notna()]

        picks_idx = picks.set_index(['date','ticker'])[['proba','rank']]
        from src.backtest import simulate_trades
        tl = simulate_trades(
            prices=pr_ohlc,
            picks=picks_idx,
            tp=params['label']['take_profit'],
            sl=params['label']['stop_loss'],
            hold=params['label']['horizon_days'],
            costs_bps=params['execution']['costs_roundtrip_bps'],
            slippage_bps=params['execution']['slippage_bps']
        )
        all_trades.append(tl)
        metrics = evaluate_trades(tl)
        metrics.update({'tr_start':str(tr_start.date()), 'te_start':str(te_start.date()), 'n_picks': int(len(picks_idx))})
        results.append(metrics)

    res = pd.DataFrame(results)
    Path('ml_swing_fullrun/data/reports').mkdir(parents=True, exist_ok=True)
    res.to_csv('ml_swing_fullrun/data/reports/backtest_summary.csv', index=False)

    # selalu tulis header walau kosong
    import pandas as _pd
    all_tl = _pd.concat(all_trades, axis=0) if len(all_trades)>0 else _pd.DataFrame(columns=['signal_date','entry_date','exit_date','ticker','ret','proba','rank'])
    if all_tl is None or all_tl.empty:
        all_tl = _pd.DataFrame(columns=['signal_date','entry_date','exit_date','ticker','ret','proba','rank'])
    all_tl.to_csv('ml_swing_fullrun/data/reports/trade_log.csv', index=False)

    print('Backtest -> ml_swing_fullrun/data/reports/backtest_summary.csv & trade_log.csv')

def run_score(asof=None, params_path='ml_swing_fullrun/config/params.yaml'):
    params = yaml.safe_load(open(params_path, encoding='utf-8'))
    prices = load_prices()
    feat = make_features(prices)
    if asof:
        feat = feat[feat['date']<=pd.to_datetime(asof)]
    lbl = make_labels_5d(prices, tp=params['label']['take_profit'], sl=params['label']['stop_loss'], horizon=params['label']['horizon_days'])
    df = feat.merge(lbl, on=['date','ticker'], how='left').dropna(subset=['y'])

    X = df.drop(columns=[c for c in ['y','open','high','low','close','adj_close','vwap'] if c in df.columns], errors='ignore').select_dtypes(include='number')
    y = df['y'].astype(int)
    model = train_xgb(X, y, params['model'])

    last_day = df['date'].max()
    X_live = df[df['date']==last_day].drop(columns=[c for c in ['y','open','high','low','close','adj_close','vwap'] if c in df.columns], errors='ignore').select_dtypes(include='number')
    proba = score_proba(model, X_live).reset_index(); proba.columns = ['row','proba']
    live_meta = df[df['date']==last_day][['date','ticker']].reset_index(drop=True)
    picks = pd.concat([live_meta, proba['proba']], axis=1).sort_values('proba', ascending=False).head(params['report']['top_n'])
    out = f"ml_swing_fullrun/data/reports/reco_{last_day.strftime('%Y%m%d')}.csv"
    picks.to_csv(out, index=False)
    print(f"Score -> {out} (asof {last_day.date()})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['backtest','score'])
    parser.add_argument('--asof', default=None)
    args = parser.parse_args()
    if args.cmd=='backtest':
        run_backtest()
    else:
        run_score(asof=args.asof)
"""
Path("ml_swing_fullrun/main.py").write_text(fixed_main, encoding="utf-8")
print("main.py ditulis ulang dengan versi fix.")


# In[8]:


import yaml
p = yaml.safe_load(open("ml_swing_fullrun/config/params.yaml", encoding="utf-8"))
p["cv"]["lookback_months"] = 12   # rolling 12 bulan
p["cv"]["test_months"] = 2        # test 2 bulan
p["report"]["top_n"] = 20         # ambil 20 teratas per hari
with open("ml_swing_fullrun/config/params.yaml","w",encoding="utf-8") as f:
    yaml.safe_dump(p, f, sort_keys=False)
print("params.yaml updated:", p["cv"], "top_n:", p["report"]["top_n"])


# In[10]:


# c) Rerun backtest
get_ipython().system('python ml_swing_fullrun/main.py backtest')


# In[11]:


get_ipython().system('python ml_swing_fullrun/main.py score')
import pandas as pd, glob
reco_files = sorted(glob.glob("ml_swing_fullrun/data/reports/reco_*.csv"))
print("Rekomendasi file:", reco_files[-1])
display(pd.read_csv(reco_files[-1]))

