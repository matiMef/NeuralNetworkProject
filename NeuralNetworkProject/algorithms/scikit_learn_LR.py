from sklearn.linear_model import LinearRegression
from utils.utils import mse_mae_test, evaluation_chart, show_features_importance

import pandas as pd
import matplotlib.pyplot as plt

def linear_regression(ds) -> list:
    line_regr = LinearRegression()
    line_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())
    
    mse_test, mae_test = mse_mae_test(ds, line_regr)
    y_pred_mlp_norm = line_regr.predict(ds.X_test)
    y_pred_mlp_dollars = y_pred_mlp_norm * ds.y_std + ds.y_mean

    show_features_importance(line_regr, "Regresja Liniowa")
    evaluation_chart(ds, y_pred_mlp_dollars)

    return mse_test, mae_test 