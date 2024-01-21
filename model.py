import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('df_for_model.csv', sep=',', low_memory=False)
df.drop_duplicates(subset=['account_id'], inplace=True)  # Use account ID instead of client ID

train, val, test = np.split(df.sample(frac=1), [int(0.8*len(df)), int(0.9*len(df))])

def get_xy(dataframe, y_label, x_labels=None):
    dataframe = copy.deepcopy(dataframe)
    if not x_labels:
        x = dataframe[[c for c in dataframe.columns if c != y_label]].values
    else:
        x = dataframe[x_labels].values.reshape(-1, len(x_labels))
    
    y = dataframe[y_label].values.reshape(-1, 1)

    return x, y

x_train, y_train = get_xy(train, 'balance_after_trans', x_labels=['account_id'])
x_val, y_val = get_xy(val, 'balance_after_trans', x_labels=['account_id'])
x_test, y_test = get_xy(test, 'balance_after_trans', x_labels=['account_id'])

model = LinearRegression()
model.fit(x_train, y_train)

# Evaluation
score_train = model.score(x_train, y_train)
print(f"Training coefficient of determination: {score_train}")

score_val = model.score(x_val, y_val)
print(f"Validation coefficient of determination: {score_val}")

score_test = model.score(x_test, y_test)
print(f"Test coefficient of determination: {score_test}")

# Visualization
plt.scatter(x_train, y_train, color='blue', label='Actual data')
plt.plot(x_train, model.predict(x_train), color='red', linewidth=2, label='Regression line')
plt.xlabel('Account ID')
plt.ylabel('Balance After Transaction')
plt.title('Linear Regression - Training Set')
plt.legend()
plt.show()

#
#   Which clients has a credit card
#

print('\n')
clients = pd.read_csv('df_merged.csv', sep=',', low_memory=False)
clients.drop_duplicates(subset=['client_id'], inplace=True)

# Count the occurrences of each card type
card_type_counts = clients['card_type'].value_counts()
print(card_type_counts)

# Plotting
x = card_type_counts.index
y = card_type_counts.values

plt.bar(x, y)
plt.xlabel('Card Type')
plt.ylabel('Number of Clients')
plt.title('Distribution of Clients by Card Type')
plt.show()

#
#   who asked loan to the bank
#

client_with_loan = []

for index, row in clients.iterrows():
    if pd.notna(row['loan_id']):
        client_with_loan.append(row['client_id'])

client_with_loan.sort()
plt.hist(client_with_loan, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('loan asked')
plt.ylabel('Number of Clients')
plt.title('Who asked loan to the Bank')
plt.show()

#
#   clients that are underage customers
#

print('\n')
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

gender = clients['client_gender'].value_counts()
plt.bar(gender.index, gender.values)
plt.xlabel('Gender')
plt.ylabel('Number of Clients')
plt.title('Number of clients per gender')
plt.show()

print('\n')

#
#   type of card that the bank offers
#

cards = clients['card_type'].unique()
print("The types of card that the bank offers are:")
for card in cards:
    if pd.notna(card):
        print(card)

