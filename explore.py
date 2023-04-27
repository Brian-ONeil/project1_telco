#####imports#####
import numpy as np
import os
import seaborn as sns
import scipy.stats as stat
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import warnings
warnings.filterwarnings("ignore")
import acquire as acq
import prepare as prep
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score

#####explore data visuals, stats tests, and models#####

def get_plot_pay(train):
    plt.figure(figsize=(12,8))
    sns.countplot(x='payment_type', hue='churn', data=train)
    plt.title('Does payment_type affect whether or not someone churns?')
    plt.xlabel('payment_type')
    plt.ylabel('Count')
    plt.show()
    
def get_stat_pay(train):
    # Create a contingency table of churn and payment_type
    contingency_table = pd.crosstab(train['churn'], train['payment_type'])

    # Perform a chi-squared test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results of the chi-squared test
    print('Chi-squared statistic:', chi2)
    print('p-value:', p)
    
def get_plot_internet(train):
    plt.figure(figsize=(12,8))
    sns.countplot(x='internet_service_type', hue='churn', data=train)
    plt.title('Does internet_service_type affect whether or not someone churns?')
    plt.xlabel('internet_service_type')
    plt.ylabel('Count')
    plt.show()
    
              
def get_stat_internet(train):
    # Create a contingency table of churn and payment_type
    contingency_table = pd.crosstab(train['churn'], train['internet_service_type'])

    # Perform a chi-squared test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results of the chi-squared test
    print('Chi-squared statistic:', chi2)
    print('p-value:', p)
    
def get_plot_contract(train):
    plt.figure(figsize=(12,8))
    sns.countplot(x='contract_type', hue='churn', data=train)
    plt.title('Does contract_type affect whether or not someone churns?')
    plt.xlabel('contract_type')
    plt.ylabel('Count')
    plt.show()
        
def get_stat_contract(train):
    # Create a contingency table of churn and payment_type
    contingency_table = pd.crosstab(train['churn'], train['contract_type'])

    # Perform a chi-squared test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results of the chi-squared test
    print('Chi-squared statistic:', chi2)
    print('p-value:', p)
    
def get_plot_tech(train):
    plt.figure(figsize=(12,8))
    sns.countplot(x='tech_support', hue='churn', data=train)
    plt.title('Does whether or not a customer has tech_support affect churn?')
    plt.xlabel('tech_support')
    plt.ylabel('Count')
    plt.show()
    
def get_stat_tech(train):
    # Create a contingency table of churn and payment_type
    contingency_table = pd.crosstab(train['churn'], train['tech_support'])

    # Perform a chi-squared test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Print the results of the chi-squared test
    print('Chi-squared statistic:', chi2)
    print('p-value:', p)
    
def get_plot_mcharges(train):
    # Visualize the relationship between churn and monthly_charges
    sns.boxplot(x='churn', y='monthly_charges', data=train)
    plt.title('Does monthly_charges affect churn?')
    plt.xlabel('Churn')
    plt.ylabel('Monthly Charges')
    plt.show()
    
def get_stat_mcharges(train):
    # Create separate dataframes for churn and non-churn customers
    churn_df = train[train['churn'] == 'Yes']
    non_churn_df = train[train['churn'] == 'No']

    # Perform a two-sample t-test for the means of monthly charges
    t_statistic, p_value = ttest_ind(churn_df['monthly_charges'],
                                     non_churn_df['monthly_charges'])

    # Print the results of the t-test
    print('T-statistic:', t_statistic)
    print('P-value:', p_value)
    
#####modeling#####

def model_prep(train, validate, test):
    # create X & y version of train/validate/test
    # where X contains the features we want to use and y is a series with just the target variable

    X_train = train.drop(columns=['customer_id', 'churn', 'tech_support', 'contract_type', 'internet_service_type', 'payment_type'])
    y_train = train.churn

    X_validate = validate.drop(columns=['customer_id', 'churn', 'tech_support', 'contract_type', 'internet_service_type', 'payment_type'])
    y_validate = validate.churn

    X_test = test.drop(columns=['customer_id', 'churn', 'tech_support', 'contract_type', 'internet_service_type', 'payment_type'])
    y_test = test.churn
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_tree(X_train, X_validate, y_train, y_validate):
    dt = DecisionTreeClassifier()

    # Define the hyperparameters to search
    params = {'max_depth': [2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5]}

    # Create a grid search object
    grid_search = GridSearchCV(dt, params, cv=5)

    # Fit the grid search object on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a decision tree model with the best hyperparameters
    dt = DecisionTreeClassifier(max_depth=best_params['max_depth'],
                            min_samples_split=best_params['min_samples_split'],
                            min_samples_leaf=best_params['min_samples_leaf'])

    # Fit the model on the training data
    dt.fit(X_train, y_train)

    # Predict the target variable for the training and validation data
    y_train_pred = dt.predict(X_train)
    y_validate_pred = dt.predict(X_validate)

    # Calculate the accuracy of the model on the training and validation data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    validate_accuracy = accuracy_score(y_validate, y_validate_pred)

    # Print the accuracy of the model on the training and validation data
    print('Training Accuracy:', train_accuracy)
    print('Validation Accuracy:', validate_accuracy)
    
def get_forest(X_train, X_validate, y_train, y_validate):
    # Create a random forest model
    rf = RandomForestClassifier()

    # Define the hyperparameters to search
    params = {'n_estimators': [50, 100, 150, 200],
              'max_depth': [2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 3, 4, 5]}

    # Create a grid search object
    grid_search = GridSearchCV(rf, params, cv=5)

    # Fit the grid search object on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a random forest model with the best hyperparameters
    rf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                max_depth=best_params['max_depth'],
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf'])

    # Fit the model on the training data
    rf.fit(X_train, y_train)

    # Predict the target variable for the training and validation data
    y_train_pred = rf.predict(X_train)
    y_validate_pred = rf.predict(X_validate)

    # Calculate the accuracy of the model on the training and validation data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    validate_accuracy = accuracy_score(y_validate, y_validate_pred)

    # Print the accuracy of the model on the training and validation data
    print('Training Accuracy:', train_accuracy)
    print('Validation Accuracy:', validate_accuracy)
    
def get_logr(X_train, X_validate, y_train, y_validate):
    # Create a logistic regression model
    lr = LogisticRegression()

    # Define the hyperparameters to search
    params = {'C': [0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

    # Create a grid search object
    grid_search = GridSearchCV(lr, params, cv=5)

    # Fit the grid search object on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a logistic regression model with the best hyperparameters
    lr = LogisticRegression(C=best_params['C'],
                            penalty=best_params['penalty'])

    # Fit the model on the training data
    lr.fit(X_train, y_train)

    # Predict the target variable for the training and validation data
    y_train_pred = lr.predict(X_train)
    y_validate_pred = lr.predict(X_validate)

    # Calculate the accuracy of the model on the training and validation data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    validate_accuracy = accuracy_score(y_validate, y_validate_pred)

    # Print the accuracy of the model on the training and validation data
    print('Training Accuracy:', train_accuracy)
    print('Validation Accuracy:', validate_accuracy)# Create a logistic regression model
    
def get_knn(X_train, X_validate, y_train, y_validate):
    # Create a KNN model
    knn = KNeighborsClassifier()

    # Define the hyperparameters to search
    params = {'n_neighbors': [3, 5, 7, 9],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}

    # Create a grid search object
    grid_search = GridSearchCV(knn, params, cv=5)

    # Fit the grid search object on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a KNN model with the best hyperparameters
    knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'],
                               weights=best_params['weights'],
                               p=best_params['p'])

    # Fit the model on the training data
    knn.fit(X_train, y_train)

    # Predict the target variable for the training and validation data
    y_train_pred = knn.predict(X_train)
    y_validate_pred = knn.predict(X_validate)

    # Calculate the accuracy of the model on the training and validation data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    validate_accuracy = accuracy_score(y_validate, y_validate_pred)

    # Print the accuracy of the model on the training and validation data
    print('Training Accuracy:', train_accuracy)
    print('Validation Accuracy:', validate_accuracy)
    
def get_logr_test(X_train, X_test, y_train, y_test):
    # Create a logistic regression model
    lr = LogisticRegression()

    # Define the hyperparameters to search
    params = {'C': [0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

    # Create a grid search object
    grid_search = GridSearchCV(lr, params, cv=5)

    # Fit the grid search object on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Create a logistic regression model with the best hyperparameters
    lr = LogisticRegression(C=best_params['C'],
                            penalty=best_params['penalty'])

    # Fit the model on the training data
    lr.fit(X_train, y_train)

    # Predict the target variable for the test data
    y_test_pred = lr.predict(X_test)

    # Calculate the accuracy of the model on the test data
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Print the accuracy of the model on the test data
    print('Test Accuracy:', test_accuracy)


            
