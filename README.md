# Analisa Saham - Swing Trading dengan Machine Learning

Project ini buat backtest strategi Swing Trading pake XGBoost + VectorBT. 
Data diambil langsung dari Yahoo Finance pake `yfinance`.

## Fitur Utama
- **Ambil Data**: Auto download data saham IHSG/US dari yfinance
- **Indikator TA**: RSI, MACD, SMA, dll pake library `ta`
- **Model ML**: Prediksi naik/turun 5 hari ke depan pake XGBoost
- **Backtest**: Simulasi profit/loss pake VectorBT
- **Visualisasi**: Grafik equity curve + signal buy/sell

## Cara Install
```bash
git clone https://github.com/[username]/Analisa-Saham.git
cd Analisa-Saham
pip install -r requirements.txt
