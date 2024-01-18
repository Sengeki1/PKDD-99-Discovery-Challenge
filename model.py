import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('df_model.csv')
del df['statement_freq']

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
cols = ['account_id', 'loan_id', 'account_district_id', 'account_date_opened', 'amount', 'duration', 'payments'] + ['status']

def scale_dataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values # features
    y = dataframe[dataframe.columns[-1]].values # label

    scaler = StandardScaler() # fit and transform
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y) # takes more of the less class and keep sampling
        # from there to increase the dataset of that smaller class so they can match 

    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y

status = ['A', 'B', 'C', 'D']
for state in status:
    z = len(train[train['status'] == state])
    print(f"Lenght of the class {state} for training: {z}")

train, x_train, y_train = scale_dataset(train, oversample=True)
# print(len(y_train))
# print(sum(y_train == 'A'))
# print(sum(y_train == 'B'))
# print(sum(y_train == 'C'))
# print(sum(y_train == 'D'))
valid, x_train, y_train = scale_dataset(valid, oversample=False)
test, x_train, y_train = scale_dataset(test, oversample=False)