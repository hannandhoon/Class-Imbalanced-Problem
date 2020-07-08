# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:37:43 2020

@author: pc
"""
#Dataset=UCI-Machine Learning Repository("Balance Scale")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Read dataset
df= pd.read_csv('H:/8th semester/Data Sciences/balance-scale.data',names=['balance', 'var1', 'var2', 'var3', 'var4'])
# Display example observations
df.head()

#Target-Variable='balance'
#4 Input-features='var1','var2','var3','var4'

#count of each class
df['balance'].value_counts()   #L=var3 * var4 < var1 * var2 , R=var3 * var4 > var1 * var2, B=var3 * var4 = var1 * var2

# Transform into binary classification
df['balance'] = [1 if b=='B' else 0 for b in df.balance] #Binary-classification
df['balance'].value_counts()  #0='Imbalanced' , 1='Balanced'

#checking imbalanced class efficiency
y = df.balance
X = df.drop('balance', axis=1) # Separate input features (X) and target variable (y)

# Train model
clf_0 = LogisticRegression().fit(X, y)

# Predict on training set
pred_y_0 = clf_0.predict(X)
print( accuracy_score(pred_y_0, y) ) #Showing 92% accuracy for one class
print( np.unique( pred_y_0 ) )  #Showing [0]  ie ignoring other class [1]

#UNDER-SAMPLING
df_majority = df[df.balance==0]
df_minority = df[df.balance==1]   # Separate majority and minority classes

# undersample majority class
df_majority_undersampled = resample(df_majority,n_samples=49,random_state=123)

# Combine minority class with downsampled majority class
df_undersampled = pd.concat([df_majority_undersampled, df_minority])

# Display new class counts
df_undersampled.balance.value_counts()
 
# Separate input features (X) and target variable (y)
y = df_undersampled.balance
X = df_undersampled.drop('balance', axis=1)
 
# Train model
clf_2 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_2 = clf_2.predict(X)
 
print( np.unique( pred_y_2 ) )

print( accuracy_score(y, pred_y_2) ) #showing 57.1% accuracy

#OVERSAMPLING

# Separate majority and minority classes
df_majority = df[df.balance==0]
df_minority = df[df.balance==1]

# oversample minority class
df_minority_oversampled = resample(df_minority,replace=True,n_samples=576,random_state=123)

# Combine majority class with oversampled minority class
df_oversampled = pd.concat([df_majority, df_minority_oversampled])

# Display new class counts
df_oversampled.balance.value_counts()

# Separate input features (X) and target variable (y)
y = df_oversampled.balance
X = df_oversampled.drop('balance', axis=1)

# Train model
clf_1 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_1 = clf_1.predict(X)

print( np.unique( pred_y_1 ) )
print( accuracy_score(y, pred_y_1) ) #showing 51.3%



