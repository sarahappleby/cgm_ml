"""
Produce the change in predictive accuracy matrices for the trained RF models as in Figure 7 of Appleby+2023.
"""

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import pickle
import sys

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import preprocessing
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_log_error, mean_squared_error
from scipy.stats import pearsonr

np.random.seed(1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)


def get_prediction_scatter(data, predicted_data, points):
    
    coords = np.transpose(np.array([data, predicted_data]))
    d_perp = np.cross(points[1] - points[0], points[0] - coords) / np.linalg.norm(points[1]-points[0])
    return np.nanstd(d_perp)


if __name__ == '__main__':

    model = sys.argv[1]
    wind = sys.argv[2]
    snap = sys.argv[3]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    plot_lines = [r'${\rm HI}\ 1215$', r'${\rm MgII}\ 2796$', r'${\rm CII}\ 1334$',
                  r'${\rm SiIII}\ 1206$', r'${\rm CIV}\ 1548$', r'${\rm OVI}\ 1031$']

    features = ['N', 'b', 'EW', 'dv', 'r_perp', 'mass', 'ssfr', 'kappa_rot'] 
    predictors = ['delta_rho', 'T', 'Z']

    features_pretty = [r'${\rm log} N$', r'$b$', r'${\rm log\ EW}$', 
                       r'${\rm d}v$', r'$f_{r200}$', r'${\rm log} M_\star$',
                       r'${\rm sSFR}$', r'$\kappa_{\rm rot}$']

    predictors_pretty = [r'${\rm log}\ \delta$', r'${\rm log}\ T$', r'${\rm log}\ Z$']

    limit_dict = {}
    limit_dict['delta_rho'] = [[0, 4], [2, 4], [2, 4], [1.5, 4], [1, 3.5], [0.5, 3.5]]
    limit_dict['T'] = [[3, 6.5], [3.5, 5], [4, 5], [4, 5], [4, 5.5], [4, 6]]
    limit_dict['Z'] = [[-4, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]

    lines = ["H1215", "MgII2796", "CII1334", "SiIII1206", "CIV1548", "OVI1031"]
    lines_short = ['HI', 'MgII', 'CII', 'SiIII', 'CIV', 'OVI']
    zsolar = [0.0134, 7.14e-4, 2.38e-3, 6.71e-4, 2.38e-3, 5.79e-3]

    model_dir = './models/'

    cmap = sns.color_palette("flare_r", as_cmap=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7.1), sharey='row', sharex='col')
    cax = plt.axes([0.15, 0.9, 0.7, 0.05])

    delta_scatter = np.zeros((len(lines), 3, len(features)))

    i = 0
    j = 0

    for l, line in enumerate(lines):

        for p, pred in enumerate(predictors):

            limits = limit_dict[pred][lines.index(line)]
            points = np.repeat(limits, 2).reshape(2, 2)

            # Load in the random forest gridsearch
            gridsearch, _, _, feature_scaler, predictor_scaler = \
                        pickle.load(open(f'{model_dir}{model}_{wind}_{snap}_{lines_short[lines.index(line)]}_lines_RF_{pred}.model', 'rb'))
            df_full = pd.read_csv(f'data/{model}_{wind}_{snap}_{line}_lines.csv') 
            train = df_full['train_mask']

            # Get the predictions from the full random forest model (using all features)
            conditions_pred = predictor_scaler.inverse_transform(np.array( gridsearch.predict(feature_scaler.transform(df_full[~train][features]))).reshape(-1, 1) )
            conditions_true = pd.DataFrame(df_full[~train],columns=[pred]).values
            scatter_orig = get_prediction_scatter(conditions_true.flatten(), conditions_pred.flatten(), points)

            for k in range(len(features)):
            
                # Iteratively choose all features but one 
                features_use = np.delete(features, k)
                idx = np.delete(np.arange(len(features)), k)

                # Scale the features and predictors to mean 0 and sigma 1 
                feature_scaler = preprocessing.StandardScaler().fit(df_full[train][features_use])
                predictor_scaler = preprocessing.StandardScaler().fit(np.array(df_full[train][pred]).reshape(-1, 1) )

                # Train a random forest model using all features but one, and the best parameters from the full grid search
                random_forest = RandomForestRegressor(n_estimators=gridsearch.best_params_['n_estimators'],
                                                      min_samples_split=gridsearch.best_params_['min_samples_split'],
                                                      min_samples_leaf=gridsearch.best_params_['min_samples_leaf'],)
                random_forest.fit(feature_scaler.transform(df_full[train][features_use]), predictor_scaler.transform(np.array(df_full[train][pred]).reshape(-1, 1) ))

                # Get the predictions from the new random forest model
                conditions_pred = predictor_scaler.inverse_transform(np.array( random_forest.predict(feature_scaler.transform(df_full[~train][features_use]))).reshape(-1, 1) )
                conditions_true = pd.DataFrame(df_full[~train],columns=[pred]).values

                conditions_pred = conditions_pred.flatten()
                conditions_true = conditions_true.flatten()

                # Compute the change in perpendicular scatter compared with the original random forest model
                delta_scatter[l][p][k] = get_prediction_scatter(conditions_true, conditions_pred, points) - scatter_orig

        scatter_use = delta_scatter[l]

        if (l == 0):
            g = sns.heatmap(scatter_use, cmap=cmap, vmin=-0.05, vmax=0.05, annot=False, ax=ax[i][j], square=True, linewidths=.5, 
                            cbar_ax=cax, cbar_kws={'label':r'$\Delta \sigma_\perp$', 'orientation':'horizontal'})
        else:
            g = sns.heatmap(scatter_use, cmap=cmap, vmin=-0.05, vmax=0.05, annot=False, ax=ax[i][j], square=True, linewidths=.5,
                            cbar=False)

        ax[i][j].set_title(plot_lines[l])

        j += 1
        if line == 'CII1334':
            i += 1
            j = 0

    for i in range(2):
        ax[i][0].set_yticklabels(predictors_pretty, rotation='horizontal', fontsize=13)
        ax[i][0].set_ylabel('Target')

    for j in range(3):
        ax[1][j].set_xticklabels(features_pretty, rotation='vertical', fontsize=13)
        ax[1][j].set_xlabel('Removed feature')

    fig.subplots_adjust(wspace=0.1, hspace=-0.35)
    plt.savefig(f'plots/{model}_{wind}_{snap}_lines_RF_delta_scatter.png')
    plt.close()
