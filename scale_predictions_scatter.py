"""
Add extra scatter to the overdensity and temperature predictions to reproduce the intrinsic scatter in the
truth data. Extra scatter is the difference between the truth and predicted data Gaussian distributions.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import pygad as pg
import pickle
import sys
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2/(2.*sigma**2))

if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]
    line = sys.argv[4]

    predictors = ['delta_rho', 'T']

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI'] 

    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]

    limit_dict = {}
    limit_dict['delta_rho'] = [[0, 4], [2, 4], [2, 4], [1.5, 4], [1, 3.5], [0.5, 3.5]]
    limit_dict['T'] = [[3, 6.5], [3.5, 5], [4, 5], [4, 5], [4, 5.5], [4, 6]]
    limit_dict['Z'] = [[-4, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    nbins = 20

    xlabels = [r'${\rm log}\ \delta_{\rm True}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm True}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm True}$']
    xlabels = [r'${\rm log}\ \delta$',
               r'${\rm log}\ (T/{\rm K})$',]
    ylabels = [r'${\rm log}\ \delta_{\rm Pred}$', 
               r'${\rm log}\ (T/{\rm K})_{\rm Pred}$', 
               r'${\rm log}\ (Z/{\rm Z}_{\odot})_{\rm Pred}$']
    x_dict = {}
    x_dict['delta_rho'] = [0.18, 0.2, 0.2, 0.18, 0.18, 0.18]
    x_dict['T'] = [0.18, 0.18, 0.18, 0.18, 0.18, 0.2]

    snapfile = f'/disk04/sapple/cgm/absorption/ml_project/data/samples/{model}_{wind}_{snap}.hdf5'
    s = pg.Snapshot(snapfile)
    redshift = s.redshift
    rho_crit = float(s.cosmology.rho_crit(z=redshift).in_units_of('g/cm**3'))
    cosmic_rho = rho_crit * float(s.cosmology.Omega_b)

    model_dir = f'/disk04/sapple/cgm/absorption/ml_project/train_spectra/models/'

    diff = {pred: None for pred in predictors}
    err = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

    data = pd.DataFrame()

    #cmap = sns.color_palette("crest", as_cmap=True)
    colors = ['#00629B', '#009988', '#CC6677']

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    for p, pred in enumerate(predictors):

        if pred == 'delta_rho':
            pred_readin = 'rho'
        else:
            pred_readin = pred

        limits = limit_dict[pred][lines.index(line)]
        points = np.repeat(limits, 2).reshape(2, 2)
        x_data = np.arange(limits[0], limits[1], 0.05)

        random_forest, features, _, feature_scaler, predictor_scaler, df_full = \
                    pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{pred_readin}.model', 'rb'))
        train = df_full['train_mask']

        test_data = df_full[~train]; del df_full
        test_data = test_data.reset_index(drop=True)
        data[f'{pred}_pred'] = pd.DataFrame(predictor_scaler.inverse_transform( np.array(random_forest.predict(feature_scaler.transform(test_data[features])).reshape(-1, 1) )),
                                  columns=[pred+'_pred'])
        data[pred] = test_data[pred_readin]

        if pred == 'delta_rho':
            data[pred] -= np.log10(cosmic_rho)
            data[f'{pred}_pred'] -= np.log10(cosmic_rho)

        hist, bin_edges = np.histogram(data[pred], bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        popt_true, _ = curve_fit(gauss, bin_centers, hist)
        sigma_true = popt_true[2]
        if sigma_true < 0:
            sigma_true *= -1

        hist, bin_edges = np.histogram(data[f'{pred}_pred'], bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        popt_pred, _ = curve_fit(gauss, bin_centers, hist)
        sigma_pred = popt_pred[2]
        if sigma_pred < 0:
            sigma_pred *= -1

        data[f'sigma_bias_{pred}'] = np.sqrt(sigma_true**2 - sigma_pred**2)
        new_scatter = np.random.normal(0, data[f'sigma_bias_{pred}'], size=len(data[pred]))
        
        delta_pred = np.abs(data[pred].values - data[f'{pred}_pred'].values) / data[pred].values 
        mask = delta_pred > 0.2
        #data[f'{pred}_new'] = data[f'{pred}_pred'].copy()
        #data[f'{pred}_new'][mask] += new_scatter[mask]
        data[f'{pred}_new'] = data[f'{pred}_pred'].copy() + new_scatter

        diff[pred] = np.array(data[pred]) - np.array(data[f'{pred}_new'])
        coords = np.transpose(np.array([data[pred], data[f'{pred}_new']]))
        d_perp = np.cross(points[1] - points[0], points[0] - coords) / np.linalg.norm(points[1]-points[0])

        scores = {}
        scores['Predictor'] = pred
        scores['Pearson'] = round(pearsonr(data[pred], data[pred+'_new'])[0], 5)
        scores['sigma_perp'] = round(np.nanstd(d_perp), 5)
        for _scorer in [r2_score, explained_variance_score, mean_squared_error]:
            scores[_scorer.__name__] = float(_scorer(data[pred],
                                               data[f'{pred}_new'], multioutput='raw_values'))
        err = err.append(scores, ignore_index=True)

        dx = (limits[1] - limits[0]) / nbins
        bins = np.arange(limits[0], limits[1]+dx, dx)
   
        mask = (data[pred] > bins[0]) & (data[pred] < bins[-1])
        diff_within = np.sum(diff[pred] < 0.2) / len(diff[pred])

        ax[p].hist(data[pred], bins=bins, stacked=True, density=True, color=colors[0], ls='-', lw=1.25, histtype='step',
                   label='Original Data')
        ax[p].hist(data[f'{pred}_pred'], bins=bins, stacked=True, density=True, color=colors[1], ls='-', lw=1.25, histtype='step',
                   label='Prediction')
        ax[p].hist(data[f'{pred}_new'], bins=bins, stacked=True, density=True, color=colors[2], ls='-', lw=1.25, histtype='step',
                   label='Prediction+Scatter') 

        ax[p].legend(fontsize=13.5, loc=4)
        ax[p].set_xlabel(xlabels[p])
        ax[p].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_scatter_bias.png')
    plt.close()

    data.to_csv(f'data/{model}_{wind}_{snap}_{line}_lines_scattered.csv')
