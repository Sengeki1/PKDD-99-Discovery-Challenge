import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import copy
import tensorflow as tf
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('df_for_model.csv', sep=',', low_memory=False)
df = df.drop_duplicates(subset=['balance_after_trans'])

train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def get_xy(dataframe, y_label, x_labels=None):
    dataframe = copy.deepcopy(dataframe)
    if not x_labels:
        x = dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_labels) == 1:
            x = dataframe[x_labels[0]].values.reshape(-1, 1)
        else:
            x = dataframe[x_labels].values
    
    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((x, y))

    return data, x , y


_, x_train, y_train = get_xy(train, 'balance_after_trans', x_labels=['client_id'])
_, x_train, y_train = get_xy(val, 'balance_after_trans', x_labels=['client_id'])
_, x_train, y_train = get_xy(test, 'balance_after_trans', x_labels=['client_id'])

model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print(f"coefficient of determination: {score}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

y_pred = model.predict(x_train)
print(f"predicted response:\n{y_pred}")

