# cgm_ml
Predicting the physical conditions in the CGM from absorption observables using random forests.

## Data

The CGM absorber dataset was generated for Appleby+2022 (see [cgm_physical_conditions](https://github.com/sarahappleby/cgm_physical_conditions)) 
using a sample of galaxies across a range of stellar masses and star formation rates, and with lines of sight for a range of impact parameters.
For this machine learning project we increased the data volume by (roughly) doubling the galaxy sample, so I needed to combine the original 
absorber dataset with the extra dataset: 
```
python make_dataset.py <Simba model> <Simba wind> <Simba snapshot> <pygad line>
```
where the Simba model is `m100n1024`, the simba wind is `s50`, and the Simba snapshot is `151`. The pygad line can be any of the following: H1215, 
MgII2796, CII1334, SiIII1206, CIV1548 or OVI1031.

## Random forests

I use the Scikit-Learn implementation of the random forest regressor algorithm to produce a mapping between the underlying physical conditions of 
the CGM absorbers and the observable absorber parameters.
```
python random_forest.py <simba model> <simba wind> <simba snapshot> <pygad line> <target feature>
```
where the target feature can be `delta_rho`, `T` or `Z` (overdensity, temperature or metallicity).

## Models

Trained models are saved in the `models' directory; see plotting routines for examples of how to read these.

## Distribution scaling

Since the random forest models make predictions that are in general too concentrated towards the mean, we have two methods of post-processsing the
predictions to reproduce the intrinsic scatter in the truth data.

To directly add extra scatter:
```
python scale_predictions_scatter.py <simba model> <simba wind> <simba snapshot> <pygad line>
```
To map onto the shape of the truth data:
```
python scale_predictions_trans.py <simba model> <simba wind> <simba snapshot> <pygad line>
```

## Plotting

For joint plots of predictions against truth data:
```
python plot_joint_lines.py <Simba model> <Simba wind> <Simba snapshot> <pygad line>
```
For feature importance matrices:
```
python plot_feature_importance.py <Simba model> <Simba wind> <Simba snapshot> <pygad line>
```
For feature accuracy importance:
```
python plot_delta_scatter.py <Simba model> <Simba wind> <Simba snapshot>
```
For plotting predictions in phase space:
```
python plot_phase_space.py <Simba model> <Simba wind> <Simba snapshot> <mode>
```
where mode can be `orig`, `scatter`, or `trans` and refers to the prediction data version.
