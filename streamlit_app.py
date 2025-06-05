#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import month_plot, plot_acf, plot_pacf

# Streamlit app setup
import streamlit as st
st.set_page_config(layout="wide")

# Membaca file data
@st.cache_data
def load_data():
    x = pd.read_csv('data/femaleaged20-19481981.csv', header=None)
    dates = pd.date_range(start='1948-01', periods=len(x), freq='ME')
    return pd.Series(x[0].values, index=dates)

x = load_data()

# Plot data keseluruhan
st.subheader("Tren Data Tenaga Kerja Wanita (1948-1981)")
fig, ax = plt.subplots(figsize=(10, 4))
x.plot(color='b', linewidth=1.5, ax=ax)
ax.set_xlabel("Tahun")
ax.set_ylabel("Jumlah Pekerja (dalam ribuan)")
ax.grid(True, linestyle="--", alpha=0.7)
st.pyplot(fig)

# Plot subset data
st.subheader("Subset Data (Jan 1963 - Des 1965)")
fig, ax = plt.subplots(figsize=(10, 4))
x['1963-01':'1965-12'].plot(marker='o', linestyle='-', color='k', ax=ax)
ax.set_ylabel("Jumlah Pekerja")
ax.grid(True, linestyle="--", alpha=0.7)
st.pyplot(fig)

# Plot musiman
st.subheader("Plot Musiman Data Tenaga Kerja Wanita (1948-1981)")
fig = month_plot(x)
plt.grid(True, linestyle="--", alpha=0.7)
st.pyplot(fig)

# Diferensiasi musiman dan biasa
st.subheader("Diferensiasi Musiman dan Biasa")
dDx = sm.tsa.statespace.tools.diff(x, k_diff=1, k_seasonal_diff=1, seasonal_periods=12)
st.write("5 data pertama setelah diferensiasi:", dDx.head())

fig, ax = plt.subplots(figsize=(10, 4))
dDx.plot(color='k', linewidth=1.2, ax=ax)
ax.set_xlabel("Tahun")
ax.set_ylabel("Diferensiasi Data")
ax.grid(True, linestyle="--", alpha=0.7)
st.pyplot(fig)

# Plot ACF dan PACF
st.subheader("ACF dan PACF runtun diferensiasi musiman")
dDx_clean = dDx.dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(dDx_clean, lags=36, ax=ax1)
plot_pacf(dDx_clean, lags=36, ax=ax2)
plt.tight_layout()
st.pyplot(fig)

# Uji KPSS
st.subheader("Hasil Uji KPSS")
result = sm.tsa.kpss(dDx_clean, regression='c', nlags="auto")
st.write(f"""
- Statistik KPSS: {result[0]:.4f}
- p-value: {result[1]:.4f}
- Lags used: {result[2]}
- Critical values: {result[3]}
""")