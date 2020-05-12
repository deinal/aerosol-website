from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np

def split_data(X, y):
    test_size = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def xgboost(X_train, y_train):
    model = XGBRegressor(max_depth=10)
    model.fit(X_train, y_train)
    return model

def randomforest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=X_train.shape[1], max_depth=10)
    model.fit(X_train, y_train.values.ravel())
    return model

def linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def score(model, X_test, y_test):
    predictions = model.predict(X_test)
    return r2_score(y_test, predictions)

def linear_equation(model, predictors, sf=4):
    # obtaining the model parameter values with sf significant figures
    coefs = []
    for coef in model.coef_[0]:
        coef = np.round(coef, sf-int(np.floor(np.log10(abs(coef))))-1)
        coefs.append(coef)
    
    coefs = np.array(coefs)
    intercept = np.round(model.intercept_, sf-int(np.floor(np.log10(abs(model.intercept_))))-1) 
    
    # string for the model equation
    equation = '\mathrm{log} N_{100} = '
    
    # adding the coefficients one after another
    for i in range(len(coefs)):
        if coefs[i] < 0:
            equation += ' - '
        elif i == 0:
            equation += ' '
        else:
            equation += ' + '
        
        equation += str(np.abs(coefs[i]))
        equation += '*' + predictors[i]
    
    # adding the intercept
    if intercept < 0:
        equation += ' - '
    else:
        equation += ' + '
        
    equation += str(np.abs(intercept[0]))               
    return equation