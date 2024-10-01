#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: auroraweng
"""
# ml pipeline

# This Python script requires 4 to 5 arguments:
# 1. Input CSV File: A clean PyRadiomic result file.
# 2. Output File Base Name: The base name for the output file (method name will be appended).
# 3. Target Column Name: The name of the target column, which must exist in the DataFrame.
# 4. Feature Selection Method: The method to use for feature selection possible options
#   - lasso
#   - univariate
#   - tree
#   - rfe
#   - pca
#   - mutual
# 5. (Optional) Excluded Columns: A comma-separated list of column names to be excluded. 
#    These might be columns not relevant to the analysis or categorical data (e.g., subject names).

# Usage: python3 ml_pipeline.py <input_csv> <output_base_name> <target_col> <feature_selection_method> [<excluded_col>]

# Example for GEMS dataset using Lasso for feature selection:
#   python3 ml_pipeline.py GEMS_output.csv GEMS_stat SDMT lasso ID

# Example for SNAPSHOT dataset using Univariate feature selection:
#   python3 ml_pipeline.py SNAPSHOT_output.csv SNAPSHOT_stat SDMT univariate ID

# Notice: This analysis is only suitable for numerical targets and numerical features.

import sys
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_regression,VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA

#feature slection

def remove_constant_features(X):
    """
    Remove features with zero variance, i.e., constant features.
    
    :param X: DataFrame containing the features.
    :return: DataFrame with constant features removed.
    """
    selector = VarianceThreshold()
    return pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

def corrSelect(X, y, num_features):
    """
    Select features based on correlation with the target variable.

    :param X: DataFrame containing the features
    :param y: Series or array containing the target variable
    :param num_features: Number of top features to select
    :return: DataFrame containing the selected features
    """
    # Calculate correlation with the target
    correlation_with_target = X.corrwith(y)
    
    # Sort by absolute value in descending order and select top 'num_features'
    top_features = correlation_with_target.abs().sort_values(ascending=False).head(num_features).index
    
    # Select the top features from the original DataFrame
    selected_X = X[top_features]

    return selected_X

def LassoSelect(X, y):
    """
    Perform feature selection using Lasso regression.

    :param X: DataFrame containing the features
    :param y: Series or array containing the target variable

    :return: DataFrame containing only the features selected by Lasso
    """
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply Lasso Regression
    lasso = LassoCV(cv=5, max_iter=200000, tol=0.01)
    lasso.fit(X_scaled, y)

    # Identify the features that Lasso kept (non-zero coefficients)
    selected_features = X.columns[np.where(lasso.coef_ != 0)[0]]
    return X[selected_features]


def univariateSelect(X, y, num_features):
    """
    Feature selection using univariate statistical tests
    """
    # Apply SelectKBest class to extract top 'num_features' features
    best_features = SelectKBest(score_func=f_classif, k=num_features)
    fit = best_features.fit(X, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    # Concat two dataframes for better visualization 
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Feature','Score']  # Naming the dataframe columns
    selected_features = feature_scores.nlargest(num_features, 'Score')['Feature']

    return X[selected_features]

def treeSelect(X, y, num_features):
    """
    Feature selection using tree-based feature importance
    """
    model = RandomForestRegressor()
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    selected_features = X.columns[indices[:num_features]]
    
    return X[selected_features]

def rfeSelect(X, y, num_features):
    """
    Feature selection using Recursive Feature Elimination (RFE)

    :param X: DataFrame containing the features
    :param y: Series or array containing the target variable
    :param num_features: Number of top features to select
    :return: Array of selected feature names
    """
    model = RandomForestRegressor()  # Suitable for regression tasks
    rfe = RFE(model, n_features_to_select=num_features)
    fit = rfe.fit(X, y)
    selected_features = X.columns[fit.support_]
    return X[selected_features]

def pcaSelect(X, num_components):
    """
    Feature selection using Principal Component Analysis (PCA)

    :param X: DataFrame containing the features
    :param num_components: Number of principal components to select
    :return: DataFrame with the principal components
    """
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(X)
    columns = [f'PC{i+1}' for i in range(num_components)]
    return pd.DataFrame(principal_components, columns=columns, index=X.index)

def mutualSelect(X, y, num_features):
    """
    Feature selection using Mutual Information
    """
    mi = mutual_info_regression(X, y)
    mi /= np.max(mi)  # Normalize the MI scores
    selected_features = X.columns[mi.argsort()[-num_features:][::-1]]
    return X[selected_features]

# Machine learning 
class Regressors():
    def __init__(self, X,y):
        ''' 
        Convert the given pandas dataframe into training and testing data.
        '''
        X = X.to_numpy()  
        y = y.to_numpy()               
        self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(X, y, test_size=0.2, random_state=42)
        self.outputs = []
        

    def test_regressor(self, reg, regressor_name=''):
        # Fit the regressor and extract metrics
        reg.fit(self.training_data, self.training_labels)

        # Extract the best parameters if GridSearchCV is used
        best_params = reg.best_params_ if isinstance(reg, GridSearchCV) else 'N/A'

        # Predictions for training and testing sets
        y_pred_train = reg.predict(self.training_data)
        y_pred_test = reg.predict(self.testing_data)

        # Calculate metrics for training data
        mae_train = mean_absolute_error(self.training_labels, y_pred_train)
        mse_train = mean_squared_error(self.training_labels, y_pred_train)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(self.training_labels, y_pred_train)

        # Calculate metrics for testing data
        mae_test = mean_absolute_error(self.testing_labels, y_pred_test)
        mse_test = mean_squared_error(self.testing_labels, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(self.testing_labels, y_pred_test)

        # Create a dictionary for the results
        results_dict = {
            'Model': regressor_name,
            'Best Parameters': best_params,
            'Training MAE': mae_train,
            'Training MSE': mse_train,
            'Training RMSE': rmse_train,
            'Training R2': r2_train,
            'Testing MAE': mae_test,
            'Testing MSE': mse_test,
            'Testing RMSE': rmse_test,
            'Testing R2': r2_test
        }

        # Append the results dictionary to the outputs list
        self.outputs.append(results_dict)
        
    def regressWithKNeighbors(self):
        # Code to run a K Nearest Neighbors regressor
        reg = GridSearchCV(KNeighborsRegressor(), {'n_neighbors': list(range(1, 20, 2)), 'leaf_size': list(range(5, 31, 5))}, cv=5)
        self.test_regressor(reg, 'K Nearest Neighbors')
    
    def regressWithLinear(self):
        # Linear Regression does not have hyperparameters for GridSearch in its basic form
        reg = LinearRegression()
        self.test_regressor(reg, 'Linear Regression')

    def regressWithDecisionTree(self):
        # Decision Tree Regressor with GridSearch
        reg = GridSearchCV(DecisionTreeRegressor(), 
                           {'max_depth': list(range(1, 51)), 'min_samples_split': list(range(2, 11))}, 
                           cv=5)
        self.test_regressor(reg, 'Decision Tree')

    def regressWithRandomForest(self):
        # Random Forest Regressor with GridSearch
        reg = GridSearchCV(RandomForestRegressor(), 
                           {'max_depth': list(range(1, 11)), 'min_samples_split': list(range(2, 11))}, 
                           cv=5)
        self.test_regressor(reg, 'Random Forest')

    def regressWithExtraTrees(self):
        # ExtraTrees Regressor with GridSearch
        param_grid = {
            'n_estimators': list(range(10, 101, 10)),
            'max_features': [None, 'sqrt', 'log2'] 
        }
        reg = GridSearchCV(ExtraTreesRegressor(), param_grid, cv=5)
        self.test_regressor(reg, 'ExtraTrees')

    def regressWithAdaBoost(self):
        # AdaBoost Regressor with GridSearch
        reg = GridSearchCV(AdaBoostRegressor(), 
                           {'n_estimators': list(range(10, 101, 10)), 'learning_rate': [0.01, 0.1, 1]}, 
                           cv=5)
        self.test_regressor(reg, 'AdaBoost')

    def regressWithXGBoost(self):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }
        reg = GridSearchCV(XGBRegressor(objective='reg:squarederror'), param_grid, cv=5)
        self.test_regressor(reg, 'XGBoost')

def main():
    start_time = time.time()
    if len(sys.argv) not in [5, 6]:
        print("Usage: python statistical_pipeline.py <input_csv> <output_csv> <target_col> <feature_selection_method> <excluded_col(optional)>")
        sys.exit(1)

    # Assign command-line arguments to respective variables
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    target_col = sys.argv[3]
    feature_selection_method = sys.argv[4].lower()  # Capture the feature selection method
    excluded_col = sys.argv[5] if len(sys.argv) == 6 and sys.argv[5].lower() != 'none' else None

    output_csv = f"{output_csv}_{feature_selection_method}.csv"
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    y = df[target_col].astype(int)
    excluded_col_list = excluded_col.split(',') if excluded_col else []
    if target_col not in excluded_col_list:
        excluded_col_list.append(target_col)
    X = df.drop(columns=excluded_col_list, errors='ignore', axis=1).astype(int)
    X = remove_constant_features(X)

    # Feature selection based on the method specified
    if feature_selection_method == 'lasso':
        X_selected = LassoSelect(X, y)
    elif feature_selection_method == 'univariate':
        X_selected = univariateSelect(X, y, num_features=10)
    elif feature_selection_method == 'tree':
        #X = corrSelect(X, y, num_features=100)
        X_selected = treeSelect(X, y, num_features=10)
    elif feature_selection_method == 'rfe':
        #X = corrSelect(X, y, num_features=100)
        X_selected = rfeSelect(X, y, num_features=10)
    elif feature_selection_method == 'pca':
        #X = corrSelect(X, y, num_features=100)
        X_selected = pcaSelect(X, num_components=10)
    elif feature_selection_method == 'mutual':
        #X = corrSelect(X, y, num_features=100)
        X_selected = mutualSelect(X, y, num_features=10)
    else:
        print("Invalid feature selection method.")
        sys.exit(1)

    # Print the feature selection method and the names of the selected features
    print(f"Selected Features using {feature_selection_method.capitalize()} method:")
    for feature_name in X_selected.columns:
        print(feature_name)
    print()
    models = Regressors(X_selected, y)


    # Running all the regression methods
    #print('Regressing with Linear Regression...')
    #models.regressWithLinear()
    print('Regressing with K Neighbors...')
    models.regressWithKNeighbors()
    print('Regressing with Decision Tree...')
    models.regressWithDecisionTree()
    print('Regressing with Random Forest...')
    models.regressWithRandomForest()
    print('Regressing with ExtraTrees...')
    models.regressWithExtraTrees()
    print('Regressing with AdaBoost...')
    models.regressWithAdaBoost()
    print('Regressing with XGBoost...')
    models.regressWithXGBoost()
    # End timing
    end_time = time.time()
    # Calculate total runtime
    total_time = end_time - start_time
    print("The code ran for", total_time, "seconds")        
    # Output the results to a file
    with open(output_csv, "w") as f:
        # Print the header
        print('Model,Best Parameters,Training MAE,Training MSE,Training RMSE,Training R2,Testing MAE,Testing MSE,Testing RMSE,Testing R2', file=f)
        
        # Print the rows for each model's results
        for result_dict in models.outputs:
            # Create a list to store the formatted results
            result_list = [
                result_dict.get('Model', 'N/A'),
                '"' + str(result_dict.get('Best Parameters', 'N/A')) + '"',  # Ensure best parameters are quoted if they're a string representation of a dictionary
                result_dict.get('Training MAE', 'N/A'),
                result_dict.get('Training MSE', 'N/A'),
                result_dict.get('Training RMSE', 'N/A'),
                result_dict.get('Training R2', 'N/A'),
                result_dict.get('Testing MAE', 'N/A'),
                result_dict.get('Testing MSE', 'N/A'),
                result_dict.get('Testing RMSE', 'N/A'),
                result_dict.get('Testing R2', 'N/A')
            ]
            
            # Join the list items into a comma-separated string and write to file
            print(','.join(str(item) for item in result_list), file=f)

if __name__ == "__main__":
    main()