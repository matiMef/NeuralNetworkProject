from dataset import Dataset
from from_scratch import NN_from_scratch
from scikit_learn_NN import MLP_NN
from scikit_learn_LR import linear_regression
from scikit_learn_RF import random_forest
from utils import scikit_MLP_NN_MSE_comparison, scikit_MLP_NN_MAE_comparison, MLP_NN_traning_MSE_chart, MLP_NN_traning_R2_chart

def main() -> None:
    ds = Dataset('kc_house_data.csv')

    # NN_from_scratch(ds, 1000, 0.001)

    scikit_mse_results = []
    scikit_mae_results = []
    loss_curves = []
    validation_scores = []

    scikit_tests = [
        [128, 64, 32, 'logistic', 'adam'],
        [64, 32, 16, 'logistic', 'adam'],
        [64, 32, 8, 'logistic', 'adam'],
        [128, 64, 32, 'relu', 'adam'],
        [64, 32, 16, 'relu', 'adam'],
        [64, 32, 8,  'relu', 'adam'],
        [64, 32, 8,  'relu', 'adam'],
        [64, 32, 8,  'relu', 'adam'],
        [64, 32, 16, 'logistic', 'sgd'],
        [64, 32, 8, 'logistic', 'sgd'],
        [128, 64, 32, 'relu', 'sgd'],
        [64, 32, 16, 'relu', 'sgd'],
        [64, 32, 8,  'relu', 'sgd']
    ]

    for test in scikit_tests:
        mse_result, mae_result, loss_curve, validation_score = MLP_NN(ds, test[0], test[1], test[2], test[3], test[4])
        scikit_mse_results.append(mse_result)
        scikit_mae_results.append(mae_result)
        loss_curves.append(loss_curve)
        validation_scores.append(validation_score)
    
    scikit_MLP_NN_MSE_comparison(scikit_mse_results)
    scikit_MLP_NN_MAE_comparison(scikit_mae_results)
    MLP_NN_traning_MSE_chart(scikit_tests, loss_curves)
    MLP_NN_traning_R2_chart(scikit_tests, validation_scores)
    
    linear_regression(ds)
    random_forest(ds)

if __name__ == "__main__":
    main()