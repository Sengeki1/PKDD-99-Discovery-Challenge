import pandas as pd

trans = pd.read_csv('cleaned_trans.csv', sep=',', low_memory=False)
order = pd.read_csv('cleaned_order.csv', sep=',')
trans_order = trans.merge(order, how='outer', left_on='account_id', right_on='account_id')

cleaned_client= pd.read_csv('cleaned_client.csv', sep=',')
cleaned_account= pd.read_csv('cleaned_account.csv', sep=',')
cleaned_card= pd.read_csv('cleaned_card.csv', sep=',')
cleaned_disp= pd.read_csv('cleaned_disp.csv', sep=',')
cleaned_district= pd.read_csv('cleaned_district.csv', sep=',')
cleaned_loan= pd.read_csv('cleaned_loan.csv', sep=',')

df = cleaned_client.merge(cleaned_disp, how='outer', left_on='client_id', right_on='client_id')
df = df.merge(trans_order, how='outer', left_on='account_id', right_on='account_id')
df = df.merge(cleaned_account, how='outer', left_on='account_id', right_on='account_id')
df = df.merge(cleaned_loan, how='outer', left_on='account_id', right_on='account_id')
df = df.merge(cleaned_disp, how='outer', left_on='disp_id', right_on='disp_id')
df = df.merge(cleaned_card, how='outer', left_on='disp_id', right_on='disp_id')
df = df.merge(cleaned_district, how='outer', left_on='client_district_id', right_on='district_id')

df = df.drop(['client_id_y', 'account_id_y', 'type_y'], axis=1)

df.columns = df.columns.str.replace(r'_x', '')
df.columns = df.columns.str.replace(r'_y', '')
df.columns = df.columns.str.replace(r' ', '_')

df.to_csv('df_merged.csv', index=False, sep=',')
print('export completed!') 