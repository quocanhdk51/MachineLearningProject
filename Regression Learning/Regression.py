#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


train = pd.read_csv('train.csv').drop(['ID'], axis = 1)
test = pd.read_csv('test.csv').drop(['ID'], axis = 1)


# In[3]:


trainX = train.drop(['TARGET_LifeExpectancy'], axis = 1)
trainY = train[['TARGET_LifeExpectancy']]


# In[4]:


#Regresion Methods
def linear_regression(X_train, Y_train):
    linear = LinearRegression()
    linear.fit(X_train, Y_train)
    return linear

def polynomial_regression(X_train, Y_train, n):
    poly_feat = PolynomialFeatures(degree = n, include_bias = False)
    poly_feat.fit(X_train)
    X_train_poly = poly_feat.transform(X_train)
    linear_4 = LinearRegression()
    linear_4.fit(X_train_poly, Y_train)
    return linear_4, poly_feat

def lasso_polynomial_regression(trainX, trainY, validX, validY, regPara, poly_feat):
    polyLassoReg = Lasso(alpha = regPara, normalize = True)
    polyFitTrainX = poly_feat.fit_transform(trainX)
    polyLassoReg.fit(polyFitTrainX, trainY)
    polyFitValidX = poly_feat.fit_transform(validX)
    predY = polyLassoReg.predict(polyFitValidX)
    mse = mean_squared_error(predY, validY)
    return predY, mse

def kfold_lasso(X, Y, split, lRegPara, poly_feat):
    kFlod = model_selection.KFold(n_splits = split, shuffle = False)
    final_result = []
    
    for trainIndex, validIndex in kFlod.split(X, Y):
        trainX = np.array(X.iloc[trainIndex])
        trainY = np.array(Y.iloc[trainIndex])
        validX = np.array(X.iloc[validIndex])
        validY = np.array(Y.iloc[validIndex])

        lResult = []
        lPredict = []
        
        for regPara in lRegPara:
            predY, mse = lasso_polynomial_regression(trainX, trainY, validX, validY, regPara, poly_feat)
            lResult.append(mse)
        final_result.append(lResult)
    return np.array(final_result).mean(axis=0)

def kfold_linear(X, Y, split):
    kFlod = model_selection.KFold(n_splits = split, shuffle = False)
    final_result = []
    
    for trainIndex, validIndex in kFlod.split(X, Y):
        trainX = np.array(X.iloc[trainIndex])
        trainY = np.array(Y.iloc[trainIndex])
        validX = np.array(X.iloc[validIndex])
        validY = np.array(Y.iloc[validIndex])
        linear = linear_regression(trainX, trainY)
        mse = mean_squared_error(linear.predict(validX), validY)
        final_result.append(mse)
    return np.array(final_result).mean(axis=0)

def kfold_poly(X, Y, split, deg):
    kFlod = model_selection.KFold(n_splits = split, shuffle = False)
    final_result = []
    
    for trainIndex, validIndex in kFlod.split(X, Y):
        trainX = np.array(X.iloc[trainIndex])
        trainY = np.array(Y.iloc[trainIndex])
        validX = np.array(X.iloc[validIndex])
        validY = np.array(Y.iloc[validIndex])
        linear_4, poly_feat = polynomial_regression(trainX, trainY, deg)
        mse = mean_squared_error(linear_4.predict(poly_feat.transform(validX)), validY)
        final_result.append(mse)
    return np.array(final_result).mean(axis=0)


# In[5]:





# In[11]:


lRegPara = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
#First Trial
trainX_1 = trainX[['Schooling', 'IncomeCompositionOfResources']]
predY_1, poly_feat_1 = polynomial_regression(trainX_1, trainY, 4)
linear_1 = kfold_linear(trainX_1, trainY, 5)
poly_1 = kfold_poly(trainX_1, trainY, 5, 4)
lasso_1 = kfold_lasso(trainX_1, trainY, 5, lRegPara, poly_feat_1)


# In[12]:


linear_1


# In[13]:


poly_1


# In[14]:


lasso_1


# In[15]:


#Second trial
trainX_2 = train[['IncomeCompositionOfResources', 'Schooling', 'Status', 'Alcohol', 'BMI', 'Polio', 'Diphtheria', 'GDP']]
predY_2, poly_feat_2 = polynomial_regression(trainX_2, trainY, 4)
linear_2 = kfold_linear(trainX_2, trainY, 5)
poly_2 = kfold_poly(trainX_2, trainY, 5, 4)
lasso_2 = kfold_lasso(trainX_2, trainY, 5, lRegPara, poly_feat_2)


# In[16]:


linear_2


# In[17]:


poly_2


# In[18]:


lasso_2


# In[19]:


#Third Trial
trainX_3 = train[['AdultMortality', 'AdultMortality-Male', 'AdultMortality-Female', 'HIV-AIDS', 'Thinness1-19years', 'Thinness5-9years']]
predY_3, poly_feat_3 = polynomial_regression(trainX_3, trainY, 4)
linear_3 = kfold_linear(trainX_3, trainY, 5)
poly_3 = kfold_poly(trainX_3, trainY, 5, 4)
lasso_3 = kfold_lasso(trainX_3, trainY, 5, lRegPara, poly_feat_3)


# In[20]:


linear_3


# In[21]:


poly_3


# In[22]:


lasso_3


# In[24]:


#Forth trial
linear_4 = kfold_linear(trainX, trainY, 5)
poly_4 = kfold_poly(trainX, trainY, 5, 4)
#The k-fold lasso regression facing some issues related to storage but normal lasso regression is still fine
predY_4, poly_feat_4 = polynomial_regression(trainX, trainY, 4)
polyLassoReg = Lasso(alpha = 0.01, normalize = True) #0.01 is the best value for regularisation weight
polyFitTrainX = poly_feat.fit_transform(trainX)
polyLassoReg.fit(polyFitTrainX, trainY)
polyFitTest = poly_feat.fit_transform(test)
predY = polyLassoReg.predict(polyFitTest)


# In[25]:


linear_4


# In[26]:


poly_4


# In[27]:


predY ##print out the prediction instead of MSE which is calculated when submitting to kaggle competition. The MSE is 4.81792


# In[28]:


#Fifth trial for testing Polynomial regression
trainX_5 = trainX[['Schooling']]
poly_5 = kfold_poly(trainX_5, trainY, 5, 4)
poly_5


# In[29]:


trainX_6 = trainX[['IncomeCompositionOfResources']]
poly_6 = kfold_poly(trainX_6, trainY, 5, 4)
poly_6


# In[ ]:




