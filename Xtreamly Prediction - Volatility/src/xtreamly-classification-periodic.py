# %reset -f
import os
import sys
import pandas as pd
import numpy as np
import time
import pytz
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sklearn
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, LinearSegmentedColormap, Normalize
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Union, List
from urllib3 import HTTPResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import joblib
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
pd.set_option('display.max_columns', None)
pd.set_option('future.no_silent_downcasting', True)
pd.option_context('mode.use_inf_as_na', True)
load_dotenv()
import seaborn as sns
import itertools

from settings.plot import tailwind, _style, _style_white
folder = 'Xtreamly Classification Periods'

# Global
symbols = ['BTC', 'ETH']
horizons = [1, 5, 15, 60, 240, 720, 1440] # 1 

# Forecasts
# Need to download first from https://storage.googleapis.com/xtreamly-public/df_stddev_periods_extreme.zip
df = pd.read_csv(os.path.join('..', 'data', f'df_stddev_periods_extreme.csv'))
df['_time'] = pd.to_datetime(df['_time']).dt.tz_localize(None)
df_pred = df.copy()

# =============================================================================
# Plot
# =============================================================================
period_types = {
    'midvol': {'name': 'Medium Volatility', 'color': tailwind['yellow-500']},
    'highvol': {'name': 'High Volatility', 'color': tailwind['red-500']},
    'lowvol': {'name': 'Low Volatility', 'color': tailwind['teal-500']}, 
    }
data_periods, data_periods_summary = [], []
for i_s, s in enumerate(symbols): 
    df_s = df_pred[df_pred['_symbol']==i_s].copy()
    df_s['return_1min'] = df_s['open'].shift(1)/df_s['open']-1

    df_periods = df_s.groupby(f'period_id').agg({
        '_time': ['min', 'max'],
        'open': ['first', 'last'],
        }).reset_index(drop=False)
    df_periods.columns = [f"{c[0]}_{c[1]}" for c in df_periods.columns]
    df_periods['duration'] = (df_periods['_time_max']-df_periods['_time_min']).dt.total_seconds()/3600
    df_periods['return'] = df_periods['open_last']/df_periods['open_first']-1
    df_periods = df_periods.rename(columns={f'period_id_': 'id'})
    df_periods['type'] = [v.split('_')[0] for v in df_periods['id']]

    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"{s} Timeline Volatility Classification on Price", pad=30)
    ax.set_ylabel(f"{s} Price (USD)",labelpad=20)
    for p,v in period_types.items():
        df_p = df_s[df_s['period']==p]
        for period_id in df_p['period_id'].unique()[:]:
            df_id = df_s[df_s[f'period_id']==period_id]
            ax.plot(df_id['_time'], df_id['open'], linewidth=2, color=period_types[p]['color'], alpha=0.9)
    for p in ['lowvol', 'midvol', 'highvol']:
        ax.plot([df_s['_time'].min()], [df_s['open'].iloc[0]], linewidth=2, color=period_types[p]['color'], label=period_types[p]['name'])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    ax.legend(loc='upper right', framealpha=0.5)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Timeline Price.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"{s} Timeline Volatility Classification on VWAP", pad=30)
    ax.set_ylabel(f"ETH VWAP (USD)",labelpad=20)
    ax.fill_between(df_s['_time'],0, df_s['vwap'], alpha=0.0)
    for p,v in period_types.items():
        df_p = df_s[df_s['period']==p]
        for period_id in df_p['period_id'].unique()[:]:
            df_id = df_s[df_s[f'period_id']==period_id]
            ax.fill_between(df_id['_time'], 0, df_id['vwap'], linewidth=2, color=period_types[p]['color'], alpha=0.7)
    for p in ['lowvol', 'midvol', 'highvol']:
        ax.fill_between(df_id['_time'], 0, 0, linewidth=2, color=period_types[p]['color'], alpha=0.9, label=period_types[p]['name'])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.set_yticks(ax.get_yticks())
    ax.set_xticks(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    ax.legend(loc='upper right', framealpha=0.5)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Timeline VWAP.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"{s} Density of Periods Returns", pad=30)
    ax.set_ylabel("Density", labelpad=20)
    ax.set_xlabel("% Returns on Periods", labelpad=20)
    for p in ['lowvol', 'midvol', 'highvol']:
        df_p = df_periods[df_periods['type']==p]
        sns.kdeplot(df_p['return'], color=period_types[p]['color'], label=period_types[p]['name'], fill=False, alpha=0.9, linewidth=4, ax=ax)
    ax.tick_params(axis='y', labelleft=False)
    ax.legend(loc='upper right', framealpha=0.5)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlim(-.1, .1)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Density Periods Return.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"{s} Density of Minutely Returns", pad=30)
    ax.set_ylabel("Density", labelpad=20)
    ax.set_xlabel("% Minutely Returns", labelpad=20)
    for p in ['lowvol', 'midvol', 'highvol']:
        df_p = df_s[df_s['period']==p]
        df_p = df_p[(df_p['return_1min'] >= -.1) & (df_p['return_1min'] <= .1)]
        sns.kdeplot(df_p['return_1min'], color=period_types[p]['color'], label=period_types[p]['name'], fill=False, alpha=0.9, linewidth=4, bw_adjust=0.9, ax=ax)
    ax.tick_params(axis='y', labelleft=False)
    ax.legend(loc='upper right', framealpha=0.5)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlim(-.01, .01)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Density Minutely Return.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"{s} Density of Minutely Volumes", pad=30)
    ax.set_ylabel("Density", labelpad=20)
    ax.set_xlabel("log(Minutely Volumes)", labelpad=20)
    for p in ['lowvol', 'midvol', 'highvol']:
        df_p = df_s[df_s['period']==p]
        sns.kdeplot(np.log(1+df_p['volume']), color=period_types[p]['color'], label=period_types[p]['name'], fill=False, alpha=0.9, linewidth=4, bw_adjust=0.9, ax=ax)
    ax.tick_params(axis='y', labelleft=False)
    ax.legend(loc='upper right', framealpha=0.5)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlim(.0, np.max(np.log(df_p['volume'])))
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Density Minutely Volume.png'), dpi=200)
    plt.show()
    fig.clf()
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"{s} Density of Duration", pad=30)
    ax.set_ylabel("Density", labelpad=20)
    ax.set_xlabel("log(Minutely Volumes)", labelpad=20)
    for p in ['lowvol', 'midvol', 'highvol']:
        df_p = df_s[df_s['period']==p]
        sns.kdeplot(np.log(1+df_p['volume']), color=period_types[p]['color'], label=period_types[p]['name'], fill=False, alpha=0.9, linewidth=4, bw_adjust=0.9, ax=ax)
    ax.tick_params(axis='y', labelleft=False)
    ax.legend(loc='upper right', framealpha=0.5)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlim(.0, np.max(np.log(df_p['volume'])))
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Density Duration.png'), dpi=200)
    plt.show()
    fig.clf()    
    
    _style_white()
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_title(f"{s} Periods Duration vs % Returns", pad=30)
    ax.set_ylabel("% Period Returns", labelpad=20)
    ax.set_xlabel("Duration Hours", labelpad=20)
    for p in ['lowvol', 'midvol', 'highvol']:
        df_p = df_periods[df_periods['type']==p]
        ax.scatter(df_p['duration'], df_p['return'], color=period_types[p]['color'], label=period_types[p]['name'], alpha=0.9)
    ax.legend(loc='upper right', framealpha=0.5)
    ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_ylim(-.05, .05)
    ax.set_xlim(0, df_periods['duration'].quantile(.95))
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    fig.savefig(os.path.join('results', folder, f'{s} Scatter Duration vs Returns.png'), dpi=200)
    plt.show()
    fig.clf()
    


df_periods = pd.concat(data_periods)
df_periods_summary = pd.concat(data_periods_summary)

df_periods.to_csv(os.path.join('..', 'data', f'df_stddev_periods.csv'), index=False)
df_periods_summary.to_csv(os.path.join('..', 'data', f'df_periods_summary.csv'), index=False)

# =============================================================================
# Table
# =============================================================================
period_types = {
'lowvol': {'name': 'Low', 'color': tailwind['teal-400']}, 
'midvol': {'name': 'Medium', 'color': tailwind['purple-400']},
'highvol': {'name': 'High', 'color': tailwind['red-400']},
}

df = pd.read_csv(os.path.join('..', 'data', f'df_periods_summary.csv'))
df = df[df['type'].isin(period_types.keys())]
df['type'] = df['type'].replace({k: v['name'] for k, v in period_types.items()})
df['avg_monthly_count'] = df['id_nunique']/6
cols_summary_all = {
'symbol': 'Symbol',
'type': 'Volatility',
'avg_monthly_count': 'Avg. Instances\nMonthly',
'cov_time': '% Time\nCoverage',
'duration_mean': 'Avg. Duration\nHours',
'duration_max': 'Max Duration\nHours',
# =============================================================================
# 'duration_perc10': '10th Quantile Duration Hours',
# 'return_std': 'Std. Dev. % Return ',
# 'return_mean': 'Avg. % Return',
# 'return_min': 'Min. % Return',
# 'return_max': 'Max. % Return',
# =============================================================================
'return_perc1': '1th Quantile\n% Return',
'return_perc99': '99th Quantile\n% Return',
}
df = df[cols_summary_all.keys()].rename(columns = cols_summary_all)
for c in df.columns:
    if '%' in c: df[c] = df[c].apply(lambda x: f"{x*100:.2f}%")
    if 'Instances' in c: df[c] = df[c].apply(lambda x: f"{x:.1f}")
    if 'Hours' in c: df[c] = df[c].apply(lambda x: f"{x:.1f}")
    
    
color_menu = tailwind['stone-800']
def _tbl(*arg):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     cellLoc='center', loc='center')
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(tailwind['stone-200'])
        cell.set_linewidth(2)
        cell.set_height(.2)
        if row == 0:
            cell.set_height(.25)
            cell.set_text_props(weight='semibold', color='white')
            cell.set_facecolor(color_menu)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.tight_layout(rect=[0.004, 0.004, .996, .996])
    plt.show()
    return fig
fig, ax = plt.subplots(figsize=(12,5))
fig = _tbl()
fig.savefig(os.path.join('results', folder, f'Summary.png'), bbox_inches='tight', dpi=300)    
df_summary_all.astype(str).to_excel(os.path.join(folder, f'Classification Summary.xlsx'), index=False)  # index=False to avoid writing row indices






