import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression

#
#   Training 
#


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

#
#   Linear Regression
#


model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print(f"coefficient of determination: {score}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

y_pred = model.predict(x_train)
print(f"predicted response:\n{y_pred}")


#
#   Which clients has a credit card
#
print('\n')
clients = pd.read_csv('df_merged.csv', sep=',', low_memory=False)
clients.drop_duplicates(subset=['client_id'], inplace=True)

client_with_gold = []
client_with_junior = []
client_with_classic = []

for index, row in clients.iterrows():
    # 'index' is the index of the current row, and 'row' is a Pandas Series representing the data in that row
    # Check if the value in the 'card_type' column for the current row is 'gold
    if row['card_type'] == 'gold':
        client_with_gold.append(row['client_id'])
    elif row['card_type'] == 'junior':
        client_with_junior.append(row['client_id'])
    else:
        if row['card_type'] == 'classic':
            client_with_classic.append(row['client_id'])
client_with_gold.sort()
print(f"clients with gold credit: {client_with_gold}")
print('\n')
client_with_classic.sort()
print(f"clients with classic credit: {client_with_classic}")
print('\n')
client_with_junior.sort()
print(f"clients with junior credit: {client_with_junior}")
print('\n')

#
#   who asked loan to the bank
#

client_with_loan = []

for index, row in clients.iterrows():
    if pd.notna(row['loan_id']):
        client_with_loan.append(row['client_id'])

client_with_loan.sort()
print(f"clients who asked for loan: {client_with_loan}")
print('\n')


#
#   clients that are underage customers
#

clients['client_age'].astype(int)

underage_clients = []

for index, row in clients.iterrows():
    if row['client_age'] < 18:
        underage_clients.append(row['client_id'])

underage_clients.sort()
print(f"Clients that are underage, are: {underage_clients}")
print('\n')

#
#   Number of clients per gender
#
female = 0
male = 0
for index, row in clients.iterrows():
    if row['client_gender'] == 'F':
        female += 1
    elif row['client_gender'] == 'M':
        male += 1

print(f"Total of female clients are: {female}")
print(f"Total of male clients are: {male}")
print('\n')

#
#   type of card that the bank offers
#

cards = clients['card_type'].unique()
print("The types of card that the bank offers are:")
for card in cards:
    if pd.notna(card):
        print(card)

