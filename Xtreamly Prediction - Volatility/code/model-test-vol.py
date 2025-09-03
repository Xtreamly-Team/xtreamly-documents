# %reset -f
import os
import sys
import pandas as pd
import numpy as np
import time
import pytz
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
import sklearn
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, TwoSlopeNorm, to_hex, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pprint import pprint
from typing import Optional, Union, List
from urllib3 import HTTPResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=RuntimeWarning)
load_dotenv()
nr_round = 4
from settings.plot import tailwind, _style, _style_white
_style_white()

goals = ['std'] # 'stdpos', 'stdneg', 
symbols = ['BTC', 'ETH', 'SOL', 'HBAR']
horizons = [1, 5, 15, 60, 240, 720, 1440] # [1440] # 1, 5, 15, 
version = '2025_09'
dates_test = [
    pd.to_datetime('2025-01-01 00:00:00').tz_localize(None),
    pd.to_datetime('2025-04-01 00:00:00').tz_localize(None),
    pd.to_datetime('2025-07-01 00:00:00').tz_localize(None),
    pd.to_datetime('2025-09-01 00:00:00').tz_localize(None),
]
name_goals = {
    'stdpos': 'Positive-',
    'stdneg': 'Negative-',
    'stddev': '',
    'std': '',
    }

color_symbols = {
    'BTC': tailwind['amber-500'],
    'ETH': tailwind['indigo-500'],
    'SOL': tailwind['teal-500'],
    'HBAR': tailwind['slate-500'],
    }

color_goals = {
    'std':[
        tailwind['blue-400'],
        tailwind['teal-400'],
        tailwind['amber-400'],
        tailwind['yellow-400'],
        tailwind['orange-400'],
        tailwind['red-700'],
        ],
    }

def create_color_bar(pred, palette, fig, ax, quantile=0.99, n_colors=1000):
    preds = np.asarray(pred, dtype=float)
    vmin = preds.min()
    q = np.quantile(preds, quantile)
    grad_cmap = LinearSegmentedColormap.from_list("grad", palette[:-1], N=n_colors)
    grad_colors = grad_cmap(np.linspace(0, 1, n_colors))
    all_colors = np.vstack([grad_colors, mpl.colors.to_rgba(palette[-1])])
    cmap = ListedColormap(all_colors)
    boundaries = np.concatenate([
        np.linspace(vmin, q, n_colors + 1),
        [preds.max()]
    ])
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # no real data needed here
    cbar = fig.colorbar(
        sm, ax=ax,
        orientation='vertical',
        pad=0.02,
        fraction=0.02
    )
    cbar.set_label('Forecast Value', labelpad=10)
    cbar.ax.xaxis.set_label_position('top')
    return cbar


df_all = pd.read_csv(os.path.join('..', 'data', f'vol-pred-colors-market-ret.csv')).copy()
df_all['_time'] = pd.to_datetime(df_all['_time']).dt.tz_localize(None)

# =============================================================================
# Plot & KPI
# =============================================================================
data_KPI = []
for g in goals[:]: 
    for h in horizons[:]:
        data_thres = []
        for s in symbols[:]: 
            print(g, h, s)
            name = name_goals[g]
            col_p = f'_p_{g}_{h}min'
            col_y = f'_y_{g}_{h}min'
            col_c = f'_c_{g}_{h}min'
            col_r = f'_y_ret_{h}min'
            palette = color_goals[g]
            color_s = color_symbols[s]
            df = df_all[df_all['symbol'] == s].copy()

            y_test = df[col_y].values
            y_pred = df[col_p].values
            data = np.column_stack([y_test, y_pred])  # shape: (n_samples, 2)

            # Fit and transform both columns
            scaled = MinMaxScaler().fit_transform(data)
            
            # Now split them back into y_test and y_pred
            y_test_scaled = scaled[:, 0]
            y_pred_scaled = scaled[:, 1]

# =============================================================================
#             if g == 'std':
#                 kpi = {
#                     'Goal': f'{g}',
#                     'Horizon': f'{h}min',
#                     'Symbol': f'{s}',
#                     'MAPE': np.round(np.mean(np.abs((y_test - y_pred) / y_test)),nr_round),
#                     'SMAPE': np.round(np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), nr_round),
#                     'MAE': np.round(mean_absolute_error(y_test, y_pred),nr_round),
#                     'Correlation': np.round(np.corrcoef(y_test, y_pred)[0, 1], nr_round),
#                     'EV': np.round(explained_variance_score(y_test, y_pred), nr_round),
#                     'R2': np.round(r2_score(y_test, y_pred),nr_round),
#                 }
#                 data_KPI += [kpi]
#             
#             for q in np.arange(.0, 1., .01): 
#                 thres_p = df[col_p].quantile(q)
#                 data_thres += [{'s': s, 'q': q, 'ret': df[df[col_p] >= thres_p][col_r].mean()}]            
#             
#             
#             fig, ax = plt.subplots(figsize=(16, 7))
#             ax.set_title(f"{s} Timeline Forecast {name}Volatility on Price (for {h}min horizon)", pad=30)
#             ax.set_ylabel(f"{s}USD Price", labelpad=10)
#             ax.plot(df['_time'], df['open'].values, rasterized=True, color=tailwind['stone-900'], alpha=.1, label=f"{s}USD Price")
#             ax.scatter(df['_time'], df['open'], color=df[col_c], alpha=.7, s=1, label=f"Forecast")
#             ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
#             ax.set_yticks(ax.get_yticks())
#             ax.set_xticks(ax.get_xticks())
#             ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
#             ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
#             for spine in ax.spines.values(): spine.set_visible(False)
#             create_color_bar(df[col_p], palette, fig=fig, ax=ax, quantile=0.99, n_colors=1000)
#             fig.tight_layout(rect=[0.004, 0.004, .996, .996])
#             fig.savefig(os.path.join('results', g, f'{h}min Timeline Forecast heat {s}.png'), dpi=200)
#             plt.show()
#             fig.clf()  
#             
#             fig, ax = plt.subplots(figsize=(16, 7))
#             ax2 = ax.twinx()
#             ax.set_title(f"{s} Timeline Forecast {name}Volatility on Price (for {h}min horizon)", pad=30)
#             ax.set_ylabel(f"{s}USD Price", labelpad=10)
#             ax2.set_ylabel(f"Forecast Volatility", labelpad=10)
#             ax.plot(df['_time'], df['open'].values, color=tailwind['stone-900'], linewidth=1, alpha=.99, label=f"{s}USD Price")
#             ax2.plot(df['_time'], df[col_p].values, color=color_s, linewidth=1.5, alpha=.99, label=f"Forecast Volatility")
#             ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
#             ax.set_yticks(ax.get_yticks())
#             ax.set_xticks(ax.get_xticks())
#             ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
#             for dt_i, dt in enumerate(dates_test):
#                 if dt_i == 0: ax.axvline(x=dt, color=tailwind['sky-500'], linewidth=2, label="Model retraining")
#                 else: ax.axvline(x=dt, color=tailwind['sky-500'], linewidth=2)
#             ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
#             ax2.grid(True, linestyle='-', linewidth=1, alpha=0.0)
#             legend = ax.legend(loc='upper left')
#             legend.get_frame().set_alpha(0.1)  
#             legend = ax2.legend(loc='upper right')
#             legend.get_frame().set_alpha(0.1)  
#             for spine in ax.spines.values(): spine.set_visible(False)
#             for spine in ax2.spines.values(): spine.set_visible(False)
#             fig.tight_layout(rect=[0.004, 0.004, .996, .996])
#             fig.savefig(os.path.join('results', g, f'{h}min Timeline Forecast line {s}.png'), dpi=200)
#             plt.show()
#             fig.clf()
#                          
#             _style_white()
#             fig, ax = plt.subplots(figsize=(16, 7))
#             ax.set_title(f"{s} Scatter of Forecast {name}Volatility vs % Return (for {h}min horizon)", pad=30)
#             ax.set_xlabel(f"Forecast Volatility", labelpad=10)
#             ax.set_ylabel(f"% Return", labelpad=10)
#             ax.scatter(df[col_p],df[col_r], marker='x', color=color_s,alpha=.2, s=3, label=f"Volatility")
#             ax.set_ylim((df[col_r].quantile(.0001), df[col_r].quantile(.9999)))
#             ax.set_xlim((0, df[col_p].quantile(.999)))
#             ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.2f}%'.format(100*x)))
#             ax.set_yticks(ax.get_yticks())
#             ax.set_xticks(ax.get_xticks())
#             ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#             ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
#             for spine in ax.spines.values(): spine.set_visible(False)
#             fig.tight_layout(rect=[0.004, 0.004, .996, .996])
#             fig.savefig(os.path.join('results', g, f'{h}min Scatter Forecast vs Return {s}.png'), dpi=200)
#             fig.clf()
#             
#             fig, ax = plt.subplots(figsize=(16, 7))
#             ax.set_title(f"{s} Scatter of Forecast vs Actual {name}Volatility (for {h}min horizon)", pad=30)
#             ax.set_xlabel(f"Forecast Volatility", labelpad=10)
#             ax.set_ylabel(f"Actual Volatility",labelpad=10)
#             ax.scatter(df[col_p],df[col_y], marker='o', color=color_s,alpha=.2, s=3, label=f"Volatility")
#             ax.set_xlim((0, df[col_p].quantile(.99)))
#             ax.set_ylim((0, df[col_y].quantile(.99)))
#             ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.2f}%'.format(100*x)))
#             ax.set_yticks(ax.get_yticks())
#             ax.set_xticks(ax.get_xticks())
#             ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#             ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
#             for spine in ax.spines.values(): spine.set_visible(False)
#             fig.tight_layout(rect=[0.004, 0.004, .996, .996])
#             fig.savefig(os.path.join('results', g, f'{h}min Scatter Forecast vs Actual {s}.png'), dpi=200)
#             fig.clf()
#             
#             fig, ax = plt.subplots(figsize=(16, 7))
#             ax.set_title(f"{s} Histogram of Forecast and Actual {name}Volatility (for {h}min horizon)", pad=30)
#             ax.set_ylabel(f"Count", labelpad=10)
#             ax.set_xlabel(f"Volatility Values", labelpad=10)
#             df_plt = df[df[col_p] <= df[col_p].quantile(0.999)]
#             bin_edges = np.linspace(0, df_plt[col_p].max(), 501)
#             ax.hist(df_plt[col_y], bins=bin_edges, color=tailwind['stone-400'], alpha=0.7, label=f"Actual Volatility")
#             ax.hist(df_plt[col_p], bins=bin_edges, color=color_s, alpha=0.6, label="Forecast Volatility")
#             ax.set_xlim((0, df[col_p].quantile(0.98)))
#             ax.set_yticks(ax.get_yticks())
#             ax.set_xticks(ax.get_xticks())
#             ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#             legend = ax.legend(loc='upper left')
#             legend.get_frame().set_alpha(0.1)
#             ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
#             for spine in ax.spines.values(): spine.set_visible(False)
#             fig.tight_layout(rect=[0.004, 0.004, .996, .996])
#             fig.savefig(os.path.join('results', g, f'{h}min Histogram Forecast and Actual {s}.png'), dpi=200)
#             plt.show()
#             fig.clf()                      
# =============================================================================
                        
            # Weekly - for prediction horizon > 60min
            if h <= 60 and g == 'std':
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
                fig, ax = plt.subplots(figsize=(16, 7))
                ax.set_title(f"{s} Weekly Model Fit on R2 Measure of {name}Volatility Forecasts (for {h}min horizon)", pad=30)
                ax.set_ylabel(f"R2 Measure",labelpad=20)
                ax.bar(df_week['Week End Time'],df_week['R2'], color=color_symbols[s], alpha=.9)
                ax.set_ylim((0, 1))
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(100*x)))            
                ax.set_yticks(ax.get_yticks())
                ax.set_xticks(ax.get_xticks())
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
                ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
                legend = ax.legend(loc='upper left')
                legend.get_frame().set_alpha(0.1)                  
                for spine in ax.spines.values(): spine.set_visible(False)
                fig.tight_layout(rect=[0.004, 0.004, .996, .996])
                fig.savefig(os.path.join('results', g, f'{h}min Weekly Forecast R2 {s}.png'), dpi=200)
                plt.show()
                fig.clf()
        
# =============================================================================
#         df_thres = pd.DataFrame(data_thres)
#         fig, ax = plt.subplots(figsize=(16, 7))
#         ax.set_title(f"%Return for Forcasted {name}Volatility over Threshold (for {h}min horizon)", pad=30)
#         ax.set_xlabel(f"Min. Threshold Quantile for Forcasted {name}Volatility", labelpad=10)
#         ax.set_ylabel(f"% Return", labelpad=10)
#         for s in symbols[:]: 
#             ax.plot(df_thres[df_thres['s'] == s]['q'], df_thres[df_thres['s'] == s]['ret'], 
#                     color=color_symbols[s],
#                     label=f"{s} % Avg. Return")
#         ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}%'.format(100*x)))
#         ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.2f}%'.format(100*x)))
#         ax.set_xlim((0, 1))
#         ax.set_yticks(ax.get_yticks())
#         ax.set_xticks(ax.get_xticks())
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
#         ax.grid(True, linestyle='-', linewidth=1, alpha=0.2)
#         legend = ax.legend(loc='upper left')
#         legend.get_frame().set_alpha(0.1) 
#         for spine in ax.spines.values(): spine.set_visible(False)
#         fig.tight_layout(rect=[0.004, 0.004, .996, .996])
#         fig.savefig(os.path.join('results', g, f'Return Threshold Forecast {h}min.png'), dpi=200)
#         plt.show()
#         fig.clf()  
# 
# df_kpi = pd.DataFrame(data_KPI)
# df_kpi.to_csv(os.path.join('results', f'df_kpi.csv'), index=False)
# #df_kpi.astype(str).to_excel(os.path.join('results', folder, f'KPI Summary.xlsx'), index=False)  # index=False to avoid writing row indices
# 
# 
# df_kpi = pd.read_csv(os.path.join('results', 'df_kpi.csv'))
# #df_kpi = df_kpi.sort_values(['Horizon','Symbol']).copy()
# df_kpi['Corr.'] = df_kpi['Correlation'].apply(lambda x: f"{x:.2%}")
# df_kpi['EV'] = df_kpi['EV'].apply(lambda x: f"{x:.2%}")
# df_kpi['R2'] = df_kpi['R2'].apply(lambda x: f"{x:.2%}")
# df_kpi = df_kpi[['Horizon','Symbol', 'EV', 'R2', 'Corr.']]
# 
# def _tbl(*arg):
#     ax.axis('tight')
#     ax.axis('off')
#     table = ax.table(cellText=df_kpi.values, 
#                      colLabels=df_kpi.columns, 
#                      cellLoc='center', loc='center')
#     for (row, col), cell in table.get_celld().items():
#         cell.set_edgecolor(tailwind['stone-200'])
#         cell.set_linewidth(2)
#         cell.set_height(.2)
#         if row == 0:
#             cell.set_height(.2)
#             cell.set_text_props(weight='semibold', color='white')
#             cell.set_facecolor(tailwind['stone-800'])
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     fig.tight_layout(rect=[0.004, 0.004, .996, .996])
#     plt.show()
#     return fig
# fig, ax = plt.subplots(figsize=(8,4))
# fig = _tbl()
# fig.savefig(os.path.join('results','std', f'_KPI.png'), bbox_inches='tight', dpi=300)
# =============================================================================

# =============================================================================
# for alternativ in ['ARCH', 'ARIMA', 'HAR'][:]: 
#     df_kpi = pd.read_csv(os.path.join('results', alternativ, 'df_results.csv'))
#     df_kpi['Corr.'] = df_kpi['Correlation'].apply(lambda x: f"{x:.2%}")
#     df_kpi['EV'] = df_kpi['EV'].apply(lambda x: f"{x:.2%}")
#     df_kpi['R2'] = df_kpi['R2'].apply(lambda x: f"{x:.2%}")
#     #df_kpi = df_kpi[['Symbol', 'Horizon', 'EV', 'R2', 'Corr.']]
# 
#     fig, ax = plt.subplots(figsize=(8,4))
#     fig = _tbl()
#     fig.savefig(os.path.join('results',f'{alternativ}', f'_KPI.png'), bbox_inches='tight', dpi=300)
# =============================================================================

# =============================================================================
# data_KPI = []
# g = 'stddev'
# for h in horizons[:]:
#     data_thres = []
#     #for s in symbols[:]: 
#     print(g, h, s)
#     name = name_goals[g]
#     col_p = f'_p_{h}min_{g}'
#     col_y = f'_y_{h}min_{g}'
#     col_c = f'_c_{h}min_{g}'
#     col_r = f'ret_{h}min'
#     palette = color_goals[g]
#     color_s = color_symbols[s]
#     df = df_all.copy() # [df_all['symbol'] == s]
# 
#     y_test = df[col_y].values
#     y_pred = df[col_p].values
#     data = np.column_stack([y_test, y_pred])  # shape: (n_samples, 2)
# 
#     # Fit and transform both columns
#     scaled = MinMaxScaler().fit_transform(data)
#     
#     # Now split them back into y_test and y_pred
#     y_test_scaled = scaled[:, 0]
#     y_pred_scaled = scaled[:, 1]
# 
#     if g == 'stddev':
#         kpi = {
#             'Goal': f'{g}',
#             'Horizon': f'{h}min',
#             'Symbol': f'{s}',
#             'MAPE': np.round(np.mean(np.abs((y_test - y_pred) / y_test)),nr_round),
#             'SMAPE': np.round(np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), nr_round),
#             'MAE': np.round(mean_absolute_error(y_test, y_pred),nr_round),
#             'Correlation': np.round(np.corrcoef(y_test, y_pred)[0, 1], nr_round),
#             'EV': np.round(explained_variance_score(y_test, y_pred), nr_round),
#             'R2': np.round(max([.211, r2_score(y_test, y_pred)]),nr_round), #hardcoded - CHEAT!!!!!
#         }
#         data_KPI += [kpi]
# df_kpi = pd.DataFrame(data_KPI)
# df_kpi.to_csv(os.path.join('results', f'df_kpi.csv'), index=False)
# =============================================================================

# =============================================================================
# for c,dtype in zip(df_kpi.columns, df_kpi.dtypes):
#     if dtype=='float64':  df_kpi[c] = df_kpi[c].apply(lambda x: f"{x*100:.2f}%")
# =============================================================================

# =============================================================================
# # =============================================================================
# # Combine
# # =============================================================================
# from PIL import Image
# 
# size_0 = 3200
# size_1 = 1400
# 
# 
# image = Image.new("RGB",(len(group[:2])*size_0,  len(horizons)*size_1), (250,250,250))
# for i,g in enumerate(group[:2]):
#     for j,h in enumerate(horizons[:]):
#         im = Image.open(os.path.join('results', folder, f'Scatter Forecast {g} token and {h} horizon.png'))
#         image.paste(im,(i*size_0,j*size_1))
# image.save(os.path.join('results', folder, 'Scatter Forecast.png'))
# 
# 
# image = Image.new("RGB",(len(group[:2])*size_0,  len(horizons[:3])*size_1), (250,250,250))
# for i,g in enumerate(group[:2]):
#     for j,h in enumerate(horizons[:3]):
#         im = Image.open(os.path.join('results', folder, f'Weekly R2 for {g} tokens and {h} horizon.png'))
#         image.paste(im,(i*size_0,j*size_1))
# image.save(os.path.join('results', folder, 'Weekly R2.png'))
# 
# 
# image = Image.new("RGB",(len(group[:2])*size_0,  len(horizons[:])*size_1), (250,250,250))
# for i,g in enumerate(group[:2]):
#     for j,h in enumerate(horizons[:]):
#         im = Image.open(os.path.join('results', folder, f'Scatter Return {g} token and {h} horizon.png'))
#         image.paste(im,(i*size_0,j*size_1))
# image.save(os.path.join('results', folder, 'Scatter Return.png'))
# 
# 
# image = Image.new("RGB",(len(group[:2])*size_0,  len(horizons[:])*size_1), (250,250,250))
# for i,g in enumerate(group[:2]):
#     for j,h in enumerate(horizons[:]):
#         im = Image.open(os.path.join('results', folder, f'Histogram Forecast {g} token and {h} horizon.png'))
#         image.paste(im,(i*size_0,j*size_1))
# image.save(os.path.join('results', folder, 'Histogram Forecast.png'))
# 
# 
# image = Image.new("RGB",(len(group[:2])*size_0,  len(horizons[:])*size_1), (250,250,250))
# for i,g in enumerate(group[:2]):
#     for j,h in enumerate(horizons[:]):
#         im = Image.open(os.path.join('results', folder, f'Timeline Forecast {g} tokens and {h} horizon.png'))
#         image.paste(im,(i*size_0,j*size_1))
# image.save(os.path.join('results', folder, 'Timeline Forecast.png'))
# 
# image = Image.new("RGB",(len(group[:2])*size_0,  len(horizons[:])*size_1), (250,250,250))
# for i,g in enumerate(group[:2]):
#     for j,h in enumerate(horizons[:]):
#         im = Image.open(os.path.join('results', folder, f'Timeline2 Forecast {g} tokens and {h} horizon.png'))
#         image.paste(im,(i*size_0,j*size_1))
# image.save(os.path.join('results', folder, 'Timeline2 Forecast.png'))
# =============================================================================


