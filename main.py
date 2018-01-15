import pandas as pd
import numpy as np
import math, datetime, time
from datetime import datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# run download_dataset.py first!
df = pd.read_csv('wiki-googl.csv', index_col='Date', parse_dates=True)

# compute new stats to use as features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (
    df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# we are only going to use those columns
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
#forecast_col = 'PCT_change'
df.fillna(-99999, inplace=True)

forecast_out = 30

X = np.array(df)
X = preprocessing.scale(X)  # normalization
y = np.array(df[forecast_col])

# training
X_train = X[:-2 * forecast_out]
y_train = y[forecast_out:-forecast_out]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X_train, y_train, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

# predict last 30 days
X_eval30 = X[-2 * forecast_out:-forecast_out]
y_eval_forecast = clf.predict(X_eval30)
# y_eval30 = X[-forecast_out:]  # expected result

df['Prediction'] = np.nan
df['Prediction'][-forecast_out:] = y_eval_forecast

# predict 30 days into future
X_pred30 = X[-forecast_out:]
y_pred_forecast = clf.predict(X_pred30)

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in y_pred_forecast:
    next_date = datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

a = df[forecast_col].plot()
p = df['Prediction'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(['Actual', 'Prediction'], loc=4)
plt.show()
