"""
Produce the phase space plots (overdensity against temperature) for the trained RF models as in Figures 8-9 of Appleby+2023.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

import numpy as np
import pickle
import pygad as pg
import pandas as pd
import seaborn as sns
import sys

from scipy.stats import pearsonr
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

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
    x = [0.73, 0.67, 0.7, 0.68, 0.68, 0.69]
    x = [0.05] * 6

    height = 6 
    ratio = 5 
    space = .2

    nbins=30
    min_delta = [0, 2, 2, 1.5, 1, 0.5]
    max_delta = [4, 4, 4, 4, 3.5, 3.5]
    min_T = [3, 3.5, 4, 4, 4, 4]
    max_T = [6, 5, 5, 5, 5.5, 6]

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    plot_dir = '/disk04/sapple/cgm/absorption/ml_project/train_spectra/plots/'
    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'
    data_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/data/'

    predictors = ['rho', 'T']

    if mode == 'orig':
        pred_str = '_pred'
        pred_label = 'Prediction'
    elif mode == 'scatter':
        pred_str = '_new'
        pred_label = 'Prediction+Scatter'
    elif mode == 'trans':
        pred_str = '_pred_trans_inv'
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
        diff = {pred: None for pred in predictors}
        err = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

        for p, pred in enumerate(predictors):

            if pred == 'rho':
                pred_use = 'delta_rho'
                points = np.repeat(np.array([min_delta[l], max_delta[l]]), 2).reshape(2, 2)
            else:
                pred_use = pred
                points = np.repeat(np.array([min_T[l], max_T[l]]), 2).reshape(2, 2)

            if mode == 'orig':
                random_forest, features, _, feature_scaler, predictor_scaler, df_full = \
                            pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{pred}.model', 'rb'))

                train = df_full['train_mask']
                data[pred] = df_full[~train][pred]
                data[f'{pred}_pred'] = predictor_scaler.inverse_transform( np.array(random_forest.predict(feature_scaler.transform(df_full[~train][features])).reshape(-1, 1) )).flatten()
        
                if pred == 'rho':
                    data[pred] -= np.log10(cosmic_rho)
                    data[f'{pred}_pred'] -= np.log10(cosmic_rho)
                    data = data.rename(columns={'rho':'delta_rho', 'rho_pred':'delta_rho_pred'})

                del df_full
            
            elif mode == 'scatter':
                data = pd.read_csv(f'{data_dir}{model}_{wind}_{snap}_{line}_lines_scattered.csv')
                data = data.rename(columns={'rho':'delta_rho', 'rho_pred':'delta_rho_pred', 'rho_new':'delta_rho_new'}) 

            elif mode == 'trans':
                data = pd.read_csv(f'{data_dir}{model}_{wind}_{snap}_{line}_lines_trans.csv')


            diff[pred] = np.array(data[pred_use]) - np.array(data[f'{pred_use}{pred_str}'])

            coords = np.transpose(np.array([data[pred_use], data[f'{pred_use}{pred_str}']]))
            d_perp = np.cross(points[1] - points[0], points[0] - coords) / np.linalg.norm(points[1]-points[0])

            scores = {}
            scores['Predictor'] = pred_use
            scores['Pearson'] = round(pearsonr(data[pred_use],data[f'{pred_use}{pred_str}'])[0], 5)
            scores['sigma_perp'] = round(np.nanstd(d_perp), 5)
            for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
                scores[_scorer.__name__] = float(_scorer(data[pred_use],
                                                   data[f'{pred_use}{pred_str}'], multioutput='raw_values'))
            err = err.append(scores, ignore_index=True)

        print(line)
        print(err)
        print('\n')

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
        im = grid.ax_joint.scatter(data[f'delta_rho{pred_str}'], data[f'T{pred_str}'], c=np.log10(data['error']), cmap=cmap, s=4, vmin=-2, vmax=1)

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
        plt.savefig(f'{plot_dir}{model}_{wind}_{snap}_{lines[l]}_deltaT_pred_{mode}.png', dpi=300)
        plt.close()

