import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from utils import MSE_MAE_summary, MSE_MAE_error

def MLP_NN(ds, size_H1, size_H2, size_H3, _activation, _solver, _max_iter, _alpha) -> list:
    MLP_regr = MLPRegressor(
        hidden_layer_sizes=(size_H1, size_H2, size_H3), # (128,64,16), (64,32,16), (64,32,8)
        activation=_activation, # logistic, relu
        solver=_solver, # adam, sgd
        alpha=_alpha,
        learning_rate_init=0.001,
        max_iter=_max_iter,          
        early_stopping=True,
        n_iter_no_change=50,    
        validation_fraction=0.1, 
        random_state=1,
        verbose=True            
    )

    MLP_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())
    
    MSE_MAE_error(ds, MLP_regr)
    mse_test, mae_test = MSE_MAE_summary(ds, MLP_regr)
    return mse_test, mae_test, MLP_regr.loss_curve_, MLP_regr.validation_scores_