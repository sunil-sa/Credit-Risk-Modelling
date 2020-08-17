#Credit risk Modelling
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from scipy import stats
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
##################### DATA PRE-PROCESSING #####################################
train_data = pd.read_csv('cs-training.csv')
test_data = pd.read_csv('cs-test.csv')
train_df = train_data.copy(deep=True)
test_df = test_data.copy(deep=True)
train_df.columns
train_df = train_df.drop(['Unnamed: 0'],axis=1)
test_df = test_df.drop(['Unnamed: 0'],axis=1)
print(train_df.isnull().sum())
#Monthly income and Dependents columns are with missing values.
# We will replace the Missing values with the median value of the column:
train_df = train_df.fillna(train_df.median())
print(train_df.isnull().sum())
#Let's check if the columns in data are of desired datatypes
print(train_df.dtypes)
print(train_df['SeriousDlqin2yrs'].unique())
train_df.rename(columns={'SeriousDlqin2yrs':'Default'},inplace = True)
test_df.rename(columns={'SeriousDlqin2yrs':'Default'},inplace = True)
print(train_df['Default'].unique())
print(train_df.head())
print(train_df['Default'].value_counts())
#Data is highly imbalanced
##################### EXPLORATORY DATA ANALYSIS ###############################
#Let's us check the patterns and relations in data
sns.boxplot(x=train_df['Default'], y=train_df['age'])
[bp] = train_df.boxplot(column=['age'],by='Default',return_type='both')
#We can see that loans are approved 
sns.boxplot(x=train_df['Default'],y=train_df['DebtRatio'])
plt.ylim(0,2)
#DebtRatio = Debt/TotalAssets
#We can see that Avg DebtRatio for default class is greater than normal one.
sns.boxplot(x=train_df['Default'],y=train_df['MonthlyIncome'])
plt.ylim(0,25000)
#Avg Monthly Income of Default class is less than normal class
sns.boxplot(x=train_df['Default'],y=train_df['NumberOfDependents'])
plt.ylim(0,20)
#It is clear that having more number of dependents increase the chances to be a default
sns.boxplot(x=train_df['Default'],y=train_df['NumberOfOpenCreditLinesAndLoans'])
plt.ylim(0,100)
#This variable has no correlation with default variable
sns.boxplot(x=train_df['Default'],y=train_df['NumberRealEstateLoansOrLines'])
plt.ylim(0,50)
#This variable has no correlation with default variable
sns.boxplot(x=train_df['Default'],y=train_df['RevolvingUtilizationOfUnsecuredLines'])
plt.ylim(0,5)
sns.heatmap(train_df.corr(), vmin=-1)
plt.show()
#Heatmap says that numberoftime3059dayspastduenotworse is highyl correlated with numberoftimes90dayslate and numberoftime6089dayspastduenotworse
#We can see in all the boxplots there are many outliers, we need to deal with this outliers.
X = train_df.drop(['Default'],axis=1)
y = train_df['Default']
#Using z score to remove all the outliers
z = np.abs(stats.zscore(X))
train_df_X = X[(z < 3).all(axis=1)] 
train_df_y = y[(z < 3).all(axis=1)]
train_df_o = pd.concat([train_df_X,train_df_y],axis=1)
sns.boxplot(x=train_df_o['Default'],y=train_df_o['NumberOfDependents'])
plt.ylim(0,2)
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(train_df_X,train_df_y,random_state=42,stratify=train_df_y)
############### Training and testing different Classification models ###########
# Logistic Regression
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
logpredict = logistic.predict(X_test)
print(roc_auc_score(y_test, logistic.predict_proba(X_test)[:,1], average = 'macro', sample_weight = None))
# Adaboost
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
print(roc_auc_score(y_test, adaboost.predict_proba(X_test)[:,1], average = 'macro', sample_weight = None))
# xgboost
xgbModel = XGBClassifier(objective='binary:logistic',seed=42)
xgbModel.fit(X_train,y_train)
xgbPred = xgbModel.predict(X_test)
print(roc_auc_score(y_test, xgbModel.predict_proba(X_test)[:,1], average = 'macro', sample_weight = None))
### Hyper-parameter tuning using RandomizedSearchCV:
param_grid = {
        "learning_rate":[0.05,0.1,0.15,0.20,0.25,0.30],
        "max_depth":[1,2,3,4,5,6,8,10,12,15],
        "min_child_weight":[1,3,5,7],
        "gamma":[0.0,0.1,0.2,0.3,0.4],
        "colsample_bytree":[0.3,0.4,0.5,0.7]}

xgb = RandomizedSearchCV(estimator = xgbModel, param_distributions = param_grid, n_iter = 10, scoring = 'roc_auc', fit_params = None,n_jobs=-1,
                       cv = 10, verbose = 2).fit(X_train, y_train)
xgb.best_estimator_
bestxgb = xgb.best_estimator_.fit(X_train, y_train)
print("The ROC score after hyper-parameter tuning")
print(roc_auc_score(y_test, bestxgb.predict_proba(X_test)[:,1], average = 'macro', sample_weight = None))
