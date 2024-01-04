import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split  
from math import sqrt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


li = []
li.append(pd.read_csv("dataset/marathon_results_2016.csv"))
li.append(pd.read_csv("dataset/marathon_results_2017.csv"))
li.append(pd.read_csv("dataset/marathon_results_2015.csv"))
df = pd.concat(li,axis=0,ignore_index=True)
def time_to_min(string):
    if string != '-':
        time_segments = string.split(':')
        hours = int(time_segments[0])
        mins = int(time_segments[1])
        sec = int(time_segments[2])
        time = hours*60 + mins + np.true_divide(sec,60)
        return time
    else:
        return -1

def gender_to_numeric(value):
    if value == 'M':
        return 0
    else:
        return 1

df['Half_min'] = df.Half.apply(lambda x: time_to_min(x))
df['Full_min'] = df['Official Time'].apply(lambda x: time_to_min(x))
df['split_ratio'] = (df['Full_min'] - df['Half_min'])/(df['Half_min'])

df_split = df[df.Half_min > 0]

df['5K_mins'] = df['5K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K'].apply(lambda x: time_to_min(x))
df['10K_mins'] = df['10K_mins'] - df['5K_mins'] 

df['15K_mins'] = df['15K'].apply(lambda x: time_to_min(x))
df['15K_mins'] = df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['20K_mins'] = df['20K'].apply(lambda x: time_to_min(x))
df['20K_mins'] = df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['25K_mins'] = df['25K'].apply(lambda x: time_to_min(x))
df['25K_mins'] = df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['30K_mins'] = df['30K'].apply(lambda x: time_to_min(x))
df['30K_mins'] = df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['35K_mins'] = df['35K'].apply(lambda x: time_to_min(x))
df['35K_mins'] = df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

df['40K_mins'] = df['40K'].apply(lambda x: time_to_min(x))
df['40K_mins'] = df['40K_mins'] -  df['35K_mins'] -df['30K_mins'] -df['25K_mins'] - df['20K_mins'] -  df['15K_mins'] - df['10K_mins'] -  df['5K_mins']

columns = ['20K_mins','15K_mins','10K_mins','5K_mins']
df['avg'] = df[columns].mean(axis = 1)
df['stdev'] = df[columns].std(axis = 1)

df_split = df[(~(df['5K'] == '-')) &(~(df['10K'] == '-'))&(~(df['15K'] == '-'))&(~(df['20K'] == '-'))&(~(df['25K'] == '-')) &(~(df['30K'] == '-')) &(~(df['35K'] == '-')) &(~(df['40K'] == '-'))]
df_split = df_split[df_split.split_ratio>0]


prediction_df = df_split[['Age','M/F', 'Half_min', 'Full_min','split_ratio','5K_mins','10K_mins','15K_mins','20K_mins','25K_mins', '30K_mins', '35K_mins','40K_mins', 'stdev']] 
prediction_df['M/F'] = prediction_df['M/F'].apply(lambda x: gender_to_numeric(x))

print(prediction_df)

#----------------------------------------------------------------------------------------


INDATA = ['Age','5K_mins','M/F']


#--------------------------------------------------------------------------------------



X_train = prediction_df[INDATA]
y_train = prediction_df['Full_min']


linear_model = LinearRegression()
linear_cv_scores = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
linear_rmse_cv = np.sqrt(-linear_cv_scores.mean())

print('Linear Regression with Cross-Validation:')
print('Average RMSE:', linear_rmse_cv)

# XGBoost Regressor with Cross-Validation
xgb_model = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.005, reg_alpha=0.1, reg_lambda=0.1)
xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
xgb_rmse_cv = np.sqrt(-xgb_cv_scores.mean())

print('\nXGBoost Regressor with Cross-Validation:')
print('Average RMSE:', xgb_rmse_cv)

#------------------------------------------------------------------------------------





traindf, testdf = train_test_split(prediction_df, test_size = 0.2,random_state=60)

X_train = traindf[INDATA]
y_train = traindf['Full_min']

X_test = testdf[INDATA]
y_test = testdf['Full_min']


XBG = XGBRegressor(n_estimators=2000, max_depth=5, learning_rate=0.005, reg_alpha=0.1, reg_lambda=0.1)
XBG.fit(X_train, y_train)

LinReg = LinearRegression()
LinReg.fit(X_train,y_train)

regression_prediction = LinReg.predict(X_test)
regression_error = regression_prediction - y_test
print('\nLinearRegression------------------------------')
print('R sqruare of regression...',LinReg.score(X_test,y_test))
print('RMSE of regression...',sqrt(mean_squared_error(y_test, regression_prediction)))


xgb_regression_prediction = XBG.predict(X_test)
xgb_regression_error = xgb_regression_prediction - y_test
print('\nXGBRegressor------------------------------')
print('Gradient Boosting Regression R Square...',XBG.score(X_test,y_test))
print('RMSE of Graident Bossting Regression...',sqrt(mean_squared_error(y_test, xgb_regression_prediction)))


sns.histplot(regression_error, bins=200, kde=False, label='Linear Regression')
sns.histplot(xgb_regression_error, bins=200, kde=False, label='XGB Regression')
plt.xlim(-50, 50)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error in minutes')
plt.ylabel('Frequency')
plt.legend(['Linear Regression','XGB Regression'], loc = 2)
plt.show()



