"""
Produce the phase space plots (overdensity against temperature) for the trained RF models as in Figures 8-9 of Appleby+2023.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import sys

from scipy.stats import pearsonr
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)


def get_dperp(data, pred, points, reference='', compare='_pred'):
    coords = np.transpose(np.array([data[f'{pred}{reference}'], data[f'{pred}{compare}']]))
    d_perp = np.cross(points[1] - points[0], points[0] - coords) / np.linalg.norm(points[1]-points[0])
    return d_perp


def get_scores(err, data, pred, points, reference='', compare='_pred',):

    scores = {}
    scores['Predictor'] = pred
    scores['Pearson'] = round(pearsonr(data[f'{pred}{reference}'], data[f'{pred}{compare}'])[0], 5)

    d_perp = get_dperp(data, pred, points, reference=reference, compare=compare)
    scores['sigma_perp'] = round(np.nanstd(d_perp), 5)

    for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
        scores[_scorer.__name__] = float(_scorer(data[f'{pred}{reference}'],
                                         data[f'{pred}{compare}'], multioutput='raw_values'))
    err = err.append(scores, ignore_index=True)

    return err


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    mode = sys.argv[4]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    Nlabels = [r'${\rm log }(N\ {\rm HI} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm MgII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CII} / {\rm cm}^{-2})$', 
               r'${\rm log }(N\ {\rm SiIII} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm CIV} / {\rm cm}^{-2})$', r'${\rm log }(N\ {\rm OVI} / {\rm cm}^{-2})$']
    x = [0.05] * 6

    height = 6 
    ratio = 5 
    space = .2

    nbins=30
    min_delta = [0, 2, 2, 1.5, 1, 0.5]
    max_delta = [4, 4, 4, 4, 3.5, 3.5]
    min_T = [3, 3.5, 4, 4, 4, 4]
    max_T = [6, 5, 5, 5, 5.5, 6]

    plot_dir = './plots/'
    model_dir = './models/'
    data_dir = './data/'

    predictors = ['delta_rho', 'T']

    if mode == 'orig': # for plotting the original predictions
        pred_str = '_pred'
        pred_label = 'Prediction'
    elif mode == 'scatter': # for plotting the predictions with added scatter
        pred_str = '_scatter'
        pred_label = 'Prediction+Scatter'
    elif mode == 'trans': #  for plotting the transformed predictions
        pred_str = '_trans_inv'
        pred_label = 'Transformed Prediction'

    hist_labels = ['Truth', pred_label]
    cmap = sns.color_palette("flare_r", as_cmap=True)
    truth_color = 'C0'
    pred_color = cmap(0.5)

    for l, line in enumerate(lines):

        ddelta = (max_delta[l] - min_delta[l]) / nbins
        delta_bins = np.arange(min_delta[l], max_delta[l]+ddelta, ddelta)
        dT = (max_T[l] - min_T[l]) / nbins
        T_bins = np.arange(min_T[l], max_T[l]+dT, dT)

        data = pd.DataFrame()       
        err = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

        for p, pred in enumerate(predictors):

            if pred == 'rho':
                points = np.repeat(np.array([min_delta[l], max_delta[l]]), 2).reshape(2, 2)
            else:
                points = np.repeat(np.array([min_T[l], max_T[l]]), 2).reshape(2, 2)

            data = pd.read_csv(f'{data_dir}{model}_{wind}_{snap}_{line}_lines_scaled.csv')
            err = get_scores(err, data, pred, points, reference='', compare=pred_str)

        print(f'{line}\n{err}\n')

        data.reset_index(drop=True, inplace=True)

        data['delta_error'] = (data['delta_rho'] - data[f'delta_rho{pred_str}']) / data['delta_rho'] 
        data['T_error'] = (data['T'] - data[f'T{pred_str}']) / data['T']
        data['error'] = np.sqrt(data['delta_error']**2 + data['T_error']**2)

        grid = sns.JointGrid('delta_rho', 'T', data=data, 
                             height=height, ratio=ratio, space=space,
                             xlim=[min_delta[l], max_delta[l]], ylim=[min_T[l], max_T[l]],)
        
        grid.ax_marg_x.hist(data['delta_rho'], bins=delta_bins, color=truth_color, 
                            density=True, histtype='step')
        grid.ax_marg_x.hist(data[f'delta_rho{pred_str}'], bins=delta_bins, color=pred_color,
                            density=True, histtype='step')
        grid.ax_marg_x.spines.left.set_visible(True)
        grid.ax_marg_x.set_yticks([0, 0.1])

        grid.ax_marg_y.hist(data['T'], bins=T_bins, color=truth_color, 
                            density=True,histtype='step', orientation='horizontal')
        grid.ax_marg_y.hist(data[f'T{pred_str}'], bins=T_bins, color=pred_color,
                            density=True,histtype='step', orientation='horizontal')
        grid.ax_marg_y.spines.bottom.set_visible(True)
        grid.ax_marg_y.set_xticks([0, 0.1])

        contour = sns.kdeplot(data=data, x='delta_rho', y='T', ax=grid.ax_joint, legend=False, cumulative=False, linewidths=1)
        im = grid.ax_joint.scatter(data[f'delta_rho{pred_str}'], data[f'T{pred_str}'], c=np.log10(data['error']), cmap=cmap, s=5, vmin=-1.25, vmax=-0.5)

        grid.set_axis_labels(xlabel=r'${\rm log }\delta$', 
                             ylabel=r'${\rm log } (T / {\rm K})$')

        #grid = sns.jointplot(data=data, x='delta_rho_pred', y=f'T_pred', 
        #                     xlim=[min_delta, max_delta], ylim=[min_T, max_T],
        #                     marginal_ticks=True, marginal_kws=dict(bins=20, fill=False, stat='probability'))
        #grid.ax_joint.collections[0].set_visible(False)
        #contour = sns.kdeplot(data=data, x='delta_rho', y='T', ax=grid.ax_joint, legend=False, cumulative=False, linewidths=1)
        #im = grid.ax_joint.scatter(data['delta_rho_pred'], data['T_pred'], c=np.log10(data['error']), cmap=cmap, s=1, vmin=-2, vmax=1)

        #grid.set_axis_labels(xlabel=r'${\rm log }\delta$', 
        #                     ylabel=r'${\rm log } (T / {\rm K})$')

        if line == 'MgII2796':
            cax = grid.figure.add_axes([0.25, .73, .5, .025])
            cbar = grid.figure.colorbar(mpl.cm.ScalarMappable(norm=grid.ax_joint.collections[-1].norm, cmap=grid.ax_joint.collections[-1].cmap),
                                        cax=cax, orientation='horizontal')
            cbar.set_label(r'${\rm log} \sigma_{\rm phase}$')
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
        
        if line == 'CII1334':
            hist_lines = [Line2D([0,1],[0,1], color=color, ls='-', lw=1) for color in [truth_color, pred_color]]
            leg = grid.ax_joint.legend(hist_lines, hist_labels, loc=1, fontsize=15)
            grid.ax_joint.add_artist(leg)

        grid.ax_joint.annotate(plot_lines[lines.index(line)], xy=(x[lines.index(line)], 0.06), xycoords='axes fraction', 
                               bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))
    
        if mode == 'scatter':
            sigma_bias_delta_rho = data["sigma_bias_delta_rho"][0]
            sigma_bias_T = data["sigma_bias_T"][0]

            sigma_annotation = r'$\sigma_\delta =$'\
                               f' {sigma_bias_delta_rho:.2f}\n'\
                               r'$\sigma_T =$'\
                               f' {sigma_bias_T:.2f}'\

            grid.ax_joint.annotate(sigma_annotation, xy=(0.76, 0.06), xycoords='axes fraction',
                                   bbox=dict(boxstyle="round", fc="w", ec='dimgrey', lw=0.75))

        plt.tight_layout()
        plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_deltaT_pred_{mode}.png', dpi=300)
        plt.close()

