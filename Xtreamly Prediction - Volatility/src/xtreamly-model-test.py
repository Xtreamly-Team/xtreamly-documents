# %reset -f
import os
import sys
import pandas as pd
import numpy as np
import time
import pytz
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sklearn
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Union, List
from urllib3 import HTTPResponse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
pd.set_option('display.max_columns', None)
load_dotenv()
nr_round = 4
from settings.plot import tailwind, _style, _style_white
folder = 'Xtreamly Volatility'

# Global
symbols = ['BTC', 'ETH']
horizons = [1, 5, 15, 60, 240, 720, 1440] # 1 
palette = [tailwind['blue-500'], tailwind['teal-500'], tailwind['amber-500'], 
           tailwind['yellow-500'], tailwind['yellow-700'], tailwind['orange-700'],]

# Forecasts
# Need to download first from https://storage.googleapis.com/xtreamly-public/df_stddev_periods_extreme.zip
df = pd.read_csv(os.path.join('..', 'data', f'df_stddev_periods_extreme.csv'))
df['_time'] = pd.to_datetime(df['_time']).dt.tz_localize(None)
df['symbol'] = df['_symbol'].replace({0: 'BTC', 1: 'ETH'})
df = df.sort_values(by=['symbol', '_time'])
df_pred = df.copy()

# Color
def _pred_color(pred, palette, extreme = True, outliers=False):
    cmap = LinearSegmentedColormap.from_list('custom_palette', palette, N=1000)
    q = np.quantile(pred, 0.99)
    norm = Normalize(vmin=np.min(pred), vmax=q)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = [to_hex(sm.to_rgba(p)) if p <= q else tailwind['red-800'] for p in pred]      
    return colors
for h in horizons:
    print(h)
    df_pred[f'{h}min_pred_color'] =_pred_color(df_pred[f'{h}min_pred'], palette)


# Plot
def create_color_bar(pred, palette, fig, ax):
    cmap = LinearSegmentedColormap.from_list('custom_palette', palette, N=1000)
    q = np.quantile(pred, 0.99)
    norm = Normalize(vmin=np.min(pred), vmax=q)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array to create the color bar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, fraction=0.02)
    cbar.set_label('Forecast Value', labelpad=10)
    cbar.ax.xaxis.set_label_position('top')

data_KPI = []
for h in horizons[:]:
    for s in symbols[:]: 
        df = df_pred[(df_pred['symbol']==s)].copy()
        df['ret'] = df['open'].shift(h)/df['open']-1
        
        col_y = f'{h}min_stddev'
        col_p = f'{h}min_pred'
        col_r = f'ret'
        
        _style_white()
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_title(f"{s}USD Timeline Volatility Forecast (for {h}min horizon)", pad=30)
        ax.set_ylabel(f"{s}USD Price", labelpad=20)
        ax.plot(df['_time'], df['open'].values, color=tailwind['stone-900'], alpha=.1, label=f"ETHUSD Price")
        ax.scatter(df['_time'], df['open'], color=df[f'{col_p}_color'], alpha=.9, s=4, label=f"Forecast")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        create_color_bar(df[col_p], palette, fig, ax)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'{s} Timeline Forecast {h}min.png'), dpi=200)
        plt.show()
        fig.clf()
            
        fig, ax = plt.subplots(figsize=(16, 6))
        ax2 = ax.twinx()
        ax.set_title(f"{s}USD Timeline Volatility Forecast (for {h}min horizon)", pad=30)
        ax.set_ylabel(f"{s}USD Price", labelpad=20)
        ax2.set_ylabel(f"Forecast Volatility", labelpad=20)
        ax.plot(df['_time'], df['open'].values, color=tailwind['stone-900'], linewidth=1.5, alpha=.99, label=f"ETHUSD Price")
        ax2.plot(df['_time'], df[col_p].values, color=tailwind['teal-600'], linewidth=1.5, alpha=.7, label=f"Forecast Volatility")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        ax2.grid(True, linestyle='-', linewidth=1, alpha=0.0)
        for spine in ax.spines.values(): spine.set_visible(False)
        for spine in ax2.spines.values(): spine.set_visible(False)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'{s} Timeline2 Forecast {h}min.png'), dpi=200)
        plt.show()
        fig.clf() 
            
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_title(f"{s} Histogram of Forecast and Actual Volatility (for {h}min horizon)", pad=30)
        ax.set_ylabel(f"Count", labelpad=20)
        ax.set_xlabel(f"Values", labelpad=20)
        df_plt = df[df[col_y] <= df[col_y].quantile(0.999)]
        bin_edges = np.linspace(0, df_plt[col_y].max(), 501)
        ax.hist(df_plt[col_y], bins=bin_edges, color=tailwind['stone-400'], alpha=0.7, label=f"Actual Volatility")
        ax.hist(df_plt[col_p], bins=bin_edges, color=tailwind['teal-500'], alpha=0.6, label="Forecast Volatility")
        ax.set_xlim((0, df[col_y].quantile(0.98)))
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        legend = ax.legend(loc='upper left')
        legend.get_frame().set_alpha(0.1)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'{s} Histogram Forecast {h}min.png'), dpi=200)
        plt.show()
        fig.clf()        
        
        _style_white()
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_title(f"{s} Scatter of Forecast vs Actual Volatility (for {h}min horizon)", pad=30)
        ax.set_ylabel(f"Actual Volatility", labelpad=20)
        ax.set_xlabel(f"Forecast  Volatility",labelpad=20)
        ax.scatter(df[col_p], df[col_y], color=tailwind['teal-600'],alpha=.3, s=3, label=f"Volatility")
        ax.set_xlim((0, df[col_p].quantile(.999)))
        ax.set_ylim((0, df[col_y].quantile(.999)))
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'{s} Scatter Forecast {h}min.png'), dpi=200)
        plt.show()
        fig.clf()

        _style_white()
        fig, ax = plt.subplots(figsize=(16, 6))
        ylim = df['ret'].abs().quantile(.999)
        ax.set_title(f"{s} Scatter of Forecast vs % Return Volatility (for {h}min horizon)", pad=30)
        ax.set_xlabel(f"Forecast Volatility", labelpad=20)
        ax.set_ylabel(f"% Return",labelpad=20)
        ax.scatter(df[col_p],df[col_r], color=tailwind['stone-500'],alpha=.3, s=5, label=f"Volatility")
        ax.set_ylim((-ylim, ylim))
        ax.set_xlim((0, df[col_p].quantile(.999)))
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.2f}%'.format(100*x)))
        ax.set_yticks(ax.get_yticks())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout(rect=[0.004, 0.004, .996, .996])
        fig.savefig(os.path.join('results', folder, f'{s} Scatter Return {h}min.png'), dpi=200)
        plt.show()
        fig.clf()
        
        if h <= 60:
            df['week_number'] = df['_time'].dt.isocalendar().week
            data_week = []
            for week in df['week_number'].unique():
                df_w = df[df['week_number'] == week]
                y_test = df_w[col_y].values
                y_pred = df_w[col_p].values
                data_week += [{
                    'Week End Time': df_w['_time'].max().strftime('%Y-%m-%d'),
                    'Week Nr': week,
                    'Count': df_w.shape[0],
                    'MAPE': np.round(np.mean(np.abs((y_test - y_pred) / y_test)),nr_round),
                    'SMAPE': np.round(np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), nr_round),
                    'MAE': np.round(mean_absolute_error(y_test, y_pred),nr_round),
                    'Corr.': np.round(np.corrcoef(y_test, y_pred)[0, 1], nr_round),
                    'EV': np.round(explained_variance_score(y_test, y_pred), nr_round),
                    'R2': np.round(r2_score(y_test, y_pred),nr_round),
                }]
            df_week = pd.DataFrame(data_week)
            
            _style_white()
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.set_title(f"{s} Weekly Volatility Model Fit on R2 Measure (for {h}min horizon)", pad=30)
            ax.set_ylabel(f"R2 Measure",labelpad=10)
            ax.bar(df_week['Week End Time'],df_week['R2'], color=tailwind['teal-500'],alpha=.9)
            ax.set_ylim((0, 1))
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(100*x)))            
            ax.set_yticks(ax.get_yticks())
            ax.set_xticks(ax.get_xticks())
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
            ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig.tight_layout(rect=[0.004, 0.004, .996, .996])
            fig.savefig(os.path.join('results', folder, f'{s} Weekly R2 {h}min.png'), dpi=200)
            plt.show()
            fig.clf()

        # KPI
        y_test = df[col_y].values
        y_pred = df[col_p].values
        data_KPI += [{
            'Horizon': f'{h}min',
            'Symbol': f'{s}',
            'MAPE': np.round(np.mean(np.abs((y_test - y_pred) / y_test)),nr_round),
            'SMAPE': np.round(np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), nr_round),
            'MAE': np.round(mean_absolute_error(y_test, y_pred),nr_round),
            'Correlation': np.round(np.corrcoef(y_test, y_pred)[0, 1], nr_round),
            'EV': np.round(explained_variance_score(y_test, y_pred), nr_round),
            'R2': np.round(r2_score(y_test, y_pred),nr_round),
            }]

df_kpi = pd.DataFrame(data_KPI)
df_kpi.to_csv(os.path.join('results', folder, f'df_kpi.csv'), index=False)


df_arima = pd.read_csv(os.path.join('results', 'ARIMA', 'df_results.csv'))
df_har = pd.read_csv(os.path.join('results', 'HAR', 'df_results.csv'))
df_arch = pd.read_csv(os.path.join('results', 'ARCH', 'df_results.csv'))
df_alt = pd.concat([df_arima, df_har, df_arch])
df_alt['Horizon'] = df_alt['Horizon'].astype(str)+'min'

df_kpi = pd.read_csv(os.path.join('results', folder, 'df_kpi.csv'))

df_KPI = df_kpi.sort_values(['Symbol']).copy()
df_KPI = df_KPI[['Symbol', 'Horizon']]
df_KPI['Xtreamly Corr.'] = df_kpi['Correlation'].apply(lambda x: f"{x:.2%}")
df_KPI['Xtreamly EV'] = df_kpi['EV'].apply(lambda x: f"{x:.2%}")
df_KPI['Xtreamly R2'] = df_kpi['R2'].apply(lambda x: f"{x:.2%}")
data_alt_r2 = []
for i,r in df_kpi.iterrows():
    data_alt_r2 += [df_alt[(df_alt['Horizon'] == r['Horizon']) & (df_alt['Symbol'] == r['Symbol'])]['R2'].max()]
df_KPI['Best alternative R2'] = [f"{x:.2%}" for x in data_alt_r2]

color_menu = tailwind['stone-800']
def _tbl(*arg):
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_KPI.values, 
                     colLabels=df_KPI.columns, 
                     cellLoc='center', loc='center')
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(tailwind['stone-200'])
        cell.set_linewidth(1)
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
fig, ax = plt.subplots(figsize=(14,3.5))
fig = _tbl()
fig.savefig(os.path.join('results', folder, f'_KPI.png'), bbox_inches='tight', dpi=300)


# =============================================================================
# Combine
# =============================================================================
from PIL import Image
size_0 = 3200
size_1 = 1200

for i,s in enumerate(symbols[:]): 
    image = Image.new("RGB",(size_0,  len(horizons)*size_1), (250,250,250))
    for j,h in enumerate(horizons[:]):
        im = Image.open(os.path.join('results', folder, f'{s} Timeline Forecast {h}min.png'))
        image.paste(im,(0,j*size_1))
    image.save(os.path.join('results', folder, f'_{s} Timeline Forecast.png'))
    
    image = Image.new("RGB",(size_0,  len(horizons)*size_1), (250,250,250))
    for j,h in enumerate(horizons[:]):
        im = Image.open(os.path.join('results', folder, f'{s} Timeline2 Forecast {h}min.png'))
        image.paste(im,(0,j*size_1))
    image.save(os.path.join('results', folder, f'_{s} Timeline2 Forecast.png'))
    
    image = Image.new("RGB",(size_0,  len(horizons)*size_1), (250,250,250))
    for j,h in enumerate(horizons[:]):
        im = Image.open(os.path.join('results', folder, f'{s} Scatter Return {h}min.png'))
        image.paste(im,(0,j*size_1))
    image.save(os.path.join('results', folder, f'_{s} Scatter Return.png'))    

    image = Image.new("RGB",(size_0,  len(horizons)*size_1), (250,250,250))
    for j,h in enumerate(horizons[:]):
        im = Image.open(os.path.join('results', folder, f'{s} Scatter Forecast {h}min.png'))
        image.paste(im,(0,j*size_1))
    image.save(os.path.join('results', folder, f'_{s} Scatter Forecast.png'))

    image = Image.new("RGB",(size_0,  len(horizons)*size_1), (250,250,250))
    for j,h in enumerate(horizons[:]):
        im = Image.open(os.path.join('results', folder, f'{s} Histogram Forecast {h}min.png'))
        image.paste(im,(0,j*size_1))
    image.save(os.path.join('results', folder, f'_{s} Histogram Forecast.png'))

    image = Image.new("RGB",(size_0,  len(np.array(horizons)[np.array(horizons)<=60])*size_1), (250,250,250))
    for j,h in enumerate(np.array(horizons)[np.array(horizons)<=60]):
        im = Image.open(os.path.join('results', folder, f'{s} Weekly R2 {h}min.png'))
        image.paste(im,(0,j*size_1))
    image.save(os.path.join('results', folder, f'_{s} Weekly R2.png'))
