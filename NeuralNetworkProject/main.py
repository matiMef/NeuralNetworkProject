from dataset import Dataset
from scikit_learn_NN import MLP_NN
from scikit_learn_LR import linear_regression

def scikit_MLP_comparison(scikit_results) -> None:
    best_result = float('inf')
    best_result_idx = 999
    for result in range (len(scikit_results)):
        if scikit_results[result] < best_result:
            best_result = scikit_results[result]
            best_result_idx = result
    print('Test: ', best_result_idx, 'Min MAE: ', best_result)

def main() -> None:
    ds = Dataset('kc_house_data.csv')

    # scikit_results = []
    # scikit_tests = [
    #     [128, 64, 32, 'logistic', 'adam'],
    #     [64, 32, 16, 'logistic', 'adam'],
    #     [64, 32, 8, 'logistic', 'adam'],
    #     [128, 64, 32, 'relu', 'adam'],
    #     [64, 32, 16, 'relu', 'adam'],
    #     [64, 32, 8,  'relu', 'adam'],
    #     [64, 32, 8,  'relu', 'adam'],
    #     [64, 32, 8,  'relu', 'adam'],
    #     [64, 32, 16, 'logistic', 'sgd'],
    #     [64, 32, 8, 'logistic', 'sgd'],
    #     [128, 64, 32, 'relu', 'sgd'],
    #     [64, 32, 16, 'relu', 'sgd'],
    #     [64, 32, 8,  'relu', 'sgd']
    # ]

    # for test in scikit_tests:
    #     result = MLP_NN(ds, test[0], test[1], test[2], test[3], test[4])
    #     scikit_results.append(result)
    
    # scikit_MLP_comparison(scikit_results)

    # linear_regression(ds)

if __name__ == "__main__":
    main()
