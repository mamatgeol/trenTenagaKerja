#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import month_plot
from PythonTsa.plot_acf_pacf import acf_pacf_fig


# In[2]:


# Membaca file data
x = pd.read_csv('femaleaged20-19481981.csv', header=None)
dates = pd.date_range(start='1948-01', periods=len(x), freq='ME')
x = pd.Series(x[0].values, index=dates)


# In[3]:


# Plot data keseluruhan
x.plot(color='b', linewidth=1.5)
plt.title("Tren Data Tenaga Kerja Wanita (1948-1981)")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Pekerja (dalam ribuan)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# In[4]:


# Plot subset data
x['1963-01':'1965-12'].plot(marker='o', linestyle='-', color='k')
plt.title("Subset Data (Jan 1963 - Des 1965)")
plt.ylabel("Jumlah Pekerja")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# In[5]:


# Plot musiman
month_plot(x)
plt.title("Plot Musiman Data Tenaga Kerja Wanita (1948-1981)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# In[6]:


# Diferensiasi musiman dan biasa
dDx = sm.tsa.statespace.tools.diff(x, k_diff=1, k_seasonal_diff=1, seasonal_periods=12)
print(dDx.head())

dDx.plot(color='k', linewidth=1.2)
plt.title("Diferensiasi Musiman dan Biasa")
plt.xlabel("Tahun")
plt.ylabel("Diferensiasi Data")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# In[7]:


# Plot ACF dan PACF
fig = acf_pacf_fig(dDx.dropna(), both=True, lag=36)
plt.suptitle("ACF dan PACF runtun diferensiasi musiman", y=0.95)
plt.tight_layout()
plt.show()


# In[8]:


# Uji KPSS
result = sm.tsa.kpss(dDx.dropna(), regression='c', nlags="auto")
print(result)

