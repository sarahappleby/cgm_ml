"""
Reproduce the intrinsic scatter in the truth data by mapping the predicted data onto the shape
of the truth data distribution. We transform the truth and predicted data onto standard Gaussians, 
and map the transformed predictions back using the inverse mapping for the truth data.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import seaborn as sns

import numpy as np
import pandas as pd
import pickle
import sys

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, kstest
from sklearn import preprocessing
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error

rng = np.random.RandomState(0)
np.random.seed(1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2/(2.*sigma**2))


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100, alpha=1.):
        cmap_list = cmap(np.linspace(minval, maxval, n))
        cmap_list[:, -1] = alpha
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                                                            cmap_list)
        return new_cmap


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
    line = sys.argv[4]

    predictors = ['delta_rho', 'T']
    # Proportion of the data to use to compute the transform mapping
    split = 0.8

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI'] 
    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]

    limit_dict = {}
    limit_dict['delta_rho'] = [[0, 4], [2, 4], [2, 4], [1.5, 4], [1, 3.5], [0.5, 3.5]]
    limit_dict['T'] = [[3, 6.5], [3.5, 5], [4, 5], [4, 5], [4, 5.5], [4, 6]]
    limit_dict['Z'] = [[-4, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
    nbins = 20

    xlabels = [r'${\rm log}\ \delta$',
               r'${\rm log}\ (T/{\rm K})$']

    model_dir = './models/'

    err_pred = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])
    err_scatter = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])
    err_trans = pd.DataFrame(columns=['Predictor', 'Pearson', 'r2_score', 'explained_variance_score', 'mean_squared_error'])

    data = pd.DataFrame()

    third = 1./3
    icolor = np.arange(0., 1+third, third)
    cmap = sns.color_palette("flare_r", as_cmap=True)
    cmap = truncate_colormap(cmap, 0., 0.95)
    color_list = [cmap(i) for i in icolor]

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))   

    for p, pred in enumerate(predictors):
        
        limits = limit_dict[pred][lines.index(line)]
        points = np.repeat(limits, 2).reshape(2, 2)
        x_data = np.arange(limits[0], limits[1], 0.05)

        # Load in the random forest and absorber data
        random_forest, features, predictor, feature_scaler, predictor_scaler = \
                    pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{pred}.model', 'rb'))
        df_full = pd.read_csv(f'data/{model}_{wind}_{snap}_{line}_lines.csv')
        train = df_full['train_mask']

        # Get the predicted physical conditions
        prediction = np.array(random_forest.predict(feature_scaler.transform(df_full[~train][features])) )
        data[f'{pred}_pred'] = predictor_scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(len(prediction))
        data[pred] = np.array(df_full[pred])[~train]
        
        err_pred = get_scores(err_pred, data, pred, reference='', compare='_pred')

        # Add extra scatter to the predictions with a width set by the difference in the truth and predicted distributions.
        # We don't do the additional scatter for metallicity because it is not normally distributed.
        if pred != 'Z':

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
            data[f'{pred}_scatter'] = data[f'{pred}_pred'].copy() + new_scatter

            err_scatter = get_scores(err_scatter, data, pred, points, reference='', compare='_scatter')

        # Map the predictions onto normal distributions
        trans_train_mask = np.random.rand(len(data)) < split
        qt = preprocessing.QuantileTransformer(output_distribution="normal", random_state=rng)
        qt.fit(data[[f'{pred}', f'{pred}_pred']][trans_train_mask])
        data[f'{pred}_trans'] = qt.transform(data[[f'{pred}', f'{pred}_pred']])[:, 1]
        
        # Inverse transform the mapped predictions using the inverse transform for the truth data.
        data[f'{pred}_trans_inv'] = qt.inverse_transform(data[[f'{pred}_trans', f'{pred}_trans']])[:, 0]
        data = data.drop(columns=[f'{pred}_trans'])

        err_trans = get_scores(err_trans, data, pred, reference='', compare='_trans_inv')

        ks_pred = kstest(data[f'{pred}_pred'], data[f'{pred}'])
        ks_scatter = kstest(data[f'{pred}_scatter'], data[f'{pred}'])
        ks_trans = kstest(data[f'{pred}_trans_inv'], data[f'{pred}'])
        
        print(pred)
        print(f'KS predictions vs truth: D={ks_pred.statistic:.4f}, pvalue={ks_pred.pvalue:.4f}')
        print(f'KS transformed predictions vs truth: D={ks_trans.statistic:.4f}, pvalue={ks_trans.pvalue:.4f}')
        print(f'KS predictions+scatter vs truth: D={ks_scatter.statistic:.4f}, pvalue={ks_scatter.pvalue:.4f}')

        print('For the original predictions:')
        print(err_pred)
        print('For the predictions with scatter:')
        print(err_scatter)
        print('For the transformed predictions:')
        print(err_trans)

        # Plotting histograms
        dx = (limits[1] - limits[0]) / nbins
        bins = np.arange(limits[0], limits[1]+dx, dx)
   
        if pred != 'Z':
            ax[p].hist(data[f'{pred}_scatter'], bins=bins, stacked=True, density=True, color=color_list[3], ls='-', lw=1.25, histtype='step',
                       label='Prediction+Scatter') 
        
        ax[p].hist(data[f'{pred}_trans_inv'], bins=bins, stacked=True, density=True, color=color_list[2], ls='-', lw=1.25, histtype='step',
                   label='Transformed')
        ax[p].hist(data[f'{pred}_pred'], bins=bins, stacked=True, density=True, color=color_list[1], ls='-', lw=1.25, histtype='step',
                   label='Prediction')
        ax[p].hist(data[pred], bins=bins, stacked=True, density=True, color=color_list[0], ls='-', lw=1.25, histtype='step',
                   label='Truth')

        if p == 0:
            ax[p].legend(fontsize=13.5, loc=2, framealpha=0.2)
            ax[p].set_ylim(0, 0.9)
            ax[0].set_xlim(-0.5, )
        ax[p].set_xlabel(xlabels[p])
        ax[p].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'plots/{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_scatter_bias_trans.png')
    plt.close()

    data.to_csv(f'data/{model}_{wind}_{snap}_{line}_lines_scaled.csv')
