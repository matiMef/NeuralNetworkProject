import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from utils import  mse_mae_test, evaluation_chart, mse_chart

def convert_r2_to_mse(r2_scores, y_val_norm):
    variance = np.var(y_val_norm)
    
    mse_scores = [(1 - r2) * variance for r2 in r2_scores]
    return mse_scores

def MLP_NN(ds, size_H1, size_H2, size_H3, _activation, _solver, _max_iter, _alpha) -> list:
    X_combined = np.vstack((ds.X_tren, ds.X_val))
    Y_combined = np.vstack((ds.Y_tren_norm, ds.Y_val_norm))

    MLP_regr = MLPRegressor(
        hidden_layer_sizes=(size_H1, size_H2, size_H3), # (128,64,16), (64,32,16), (64,32,8)
        activation=_activation, # logistic, relu
        solver=_solver, # adam, sgd
        alpha=_alpha,
        learning_rate_init=0.001,
        max_iter=_max_iter,          
        early_stopping=True,
        n_iter_no_change=50,    
        validation_fraction=0.11, 
        random_state=1,
        verbose=True            
    )

    MLP_regr.fit(X_combined, Y_combined.ravel())

    mse_test, mae_test = mse_mae_test(ds, MLP_regr)
    mse_val = convert_r2_to_mse(MLP_regr.validation_scores_, Y_combined.ravel())
    y_pred_mlp_norm = MLP_regr.predict(ds.X_test)
    y_pred_mlp_dollars = y_pred_mlp_norm * ds.y_std + ds.y_mean
    
    mse_chart(MLP_regr.loss_curve_, mse_val)
    evaluation_chart(ds, y_pred_mlp_dollars)
    
    return mse_test, mae_test, MLP_regr.loss_curve_, MLP_regr.validation_scores_

