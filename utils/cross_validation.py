
from tkinter import SE
import numpy as np
import test
import xarray as xr
from sklearn.linear_model import LinearRegression

def forward_selection(predictand, predictors_dict, test_predictand = None, 
                      test_predictors_dict = None):
    """This function selects the predictors that best explain the predictand
    using the forward selection method. Optionally computes the SE of the
    linear model using the test data.
    
    Parameters
    ----------
    predictand : numpy.array
        Predictand (training data)
    predictors_dict : dict
        Dictionary with the predictors (training data)
    test_predictand : numpy.array
        Test predictand
    test_predictors_dict : dict
        Dictionary with the test predictors
    
    Returns
    -------
    list
        List with the selected predictors
    numpy.array
        Array with the R2 values of the selected predictors
    numpy.array
        Array with the MSE values of the selected predictors
    numpy.array
        Array with the SE value of the test data
    """
    
    # output vars
    predictors_selected_list = []
    predictors_selected_r2_list = []
    predictors_selected_mse_list = []
    predictors_selected_se_list = None if test_predictand is None else []

    # reshape predictand
    y = np.reshape(predictand, (-1, 1))
    predictors_name = list(predictors_dict.keys())
    predictors_lenght = len(predictors_name)
    k = 0 # counter 
    while(k < y.size-1):
        if k == 0:
            print('Select the first predictor')
            # temporarily select the predictor at position 0
            pred_selected = 0
            x0 = np.reshape(predictors_dict[predictors_name[pred_selected]], (-1, 1))
            reg = LinearRegression().fit(x0, y)
            reg_selected = reg
            pred_selected_r2 = reg.score(x0, y)
            for i in range(1, predictors_lenght):
                # i-th predictor is the candidate
                x = np.reshape(predictors_dict[predictors_name[i]], (-1, 1))
                reg = LinearRegression().fit(x, y)
                cand_r2 = reg.score(x, y)
                # swap if necessary
                if cand_r2 > pred_selected_r2:
                    pred_selected = i
                    reg_selected = reg
                    pred_selected_r2 = cand_r2
        else:
            print(f'Select the {k}-th predictor')
            first = True # first iteration
            for i in range(predictors_lenght):
                if i in predictors_selected_list:
                    continue
                else:
                    expanded_list = predictors_selected_list + [i]
                    xvars = [predictors_dict[predictors_name[j]] for j in expanded_list]
                    x = np.array(xvars).T
                    reg = LinearRegression().fit(x, y)
                    cand_r2 = reg.score(x, y)
                    if first:
                        # temporarily select the predictor at position i
                        pred_selected = i
                        reg_selected = reg
                        pred_selected_r2 = cand_r2
                        first = False
                    else:
                        # swap if necessary
                        if cand_r2 > pred_selected_r2:
                            pred_selected = i
                            reg_selected = reg
                            pred_selected_r2 = cand_r2
        # save the selected predictor
        yhat = reg.predict(x)
        mse = np.sum((y-yhat)**2)/(y.size-k)
        print(f'Add predictor {predictors_name[pred_selected]} with R2 = {pred_selected_r2}')
        predictors_selected_list.append(pred_selected)
        predictors_selected_r2_list.append(pred_selected_r2)
        predictors_selected_mse_list.append(mse)
        
        if test_predictand is not None:
            test_predictor_values = [test_predictors_dict[predictors_name[j]] for j in predictors_selected_list] # type: ignore
            x_test = np.array(test_predictor_values).reshape(1, -1)
            yhat = reg_selected.predict(x_test)
            se = np.sum((yhat - test_predictand)**2)
            predictors_selected_se_list.append(se) # type: ignore
        k = k+1
        
        
        
    return predictors_selected_list, np.array(predictors_selected_r2_list), np.array(predictors_selected_mse_list), np.array(predictors_selected_se_list)
    
def cross_validation(predictand, predictors_dict):
    """TODO.
    
    Parameters
    ----------
    predictand : numpy.array
        Predictand
    predictors_dict : dict
        Dictionary with the predictors
    
    Returns
    -------
    numpy.array
        Array with the MSE values of each linear model
    """
    
    # output vars
    n = predictand.size
    se = np.zeros((n, n-2))*np.nan
    
    for test_index in range(n):
        print(f'Cross-validation iteration {test_index+1}/{n}')
        test_predictand = predictand[test_index]
        train_predictand = np.delete(predictand, test_index) 
        
        test_predictors_dict = {k: v[test_index] for k, v in predictors_dict.items()}
        train_predictors_dict = {k: np.delete(v, test_index) for k, v in predictors_dict.items()}
        
        ans = forward_selection(train_predictand, train_predictors_dict, 
                                test_predictand, test_predictors_dict)
        
        se[test_index,:] = ans[-1]
    mse = np.mean(se, axis=0)
    return mse