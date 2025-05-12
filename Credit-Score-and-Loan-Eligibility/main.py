import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

credit_model = joblib.load('credit_model.pkl')
credit_score_model = joblib.load('credit_score_model.pkl')
gender_encoder = joblib.load('gender_encoder.pkl')
account_type_encoder = joblib.load('account_type_encoder.pkl')
loan_history_encoder = joblib.load('loan_history_encoder.pkl')

def extract_transaction_features(tx_str):
    deposits, withdrawals = [], []
    if pd.notnull(tx_str):
        for tx in tx_str.split(','):
            if 'deposit:' in tx:
                deposits.append(float(tx.split(':')[1]))
            elif 'withdrawal:' in tx:
                withdrawals.append(float(tx.split(':')[1]))
    return pd.Series([
        sum(deposits),
        sum(withdrawals),
        len(deposits) + len(withdrawals),
        (sum(deposits) + sum(withdrawals)) / (len(deposits) + len(withdrawals) or 1)
    ])

def process_customer_data(df):
    df['account_open_date'] = pd.to_datetime(df['account_open_date'])
    df['account_years'] = datetime.now().year - df['account_open_date'].dt.year
    df[['deposit_total', 'withdrawal_total', 'num_transactions', 'avg_transaction']] = df['transactions'].apply(extract_transaction_features)

    df['gender'] = gender_encoder.transform(df['gender'])
    df['account_type'] = account_type_encoder.transform(df['account_type'])
    df['loan_history'] = loan_history_encoder.transform(df['loan_history'])

    return df

df_new = pd.read_csv('test_data.csv')
df_new = process_customer_data(df_new)

credit_features = ['account_balance', 'account_years', 'loan_history', 'account_type', 'num_transactions', 'avg_transaction']
df_new['credit_score'] = credit_score_model.predict(df_new[credit_features])

rf_features = credit_features + ['gender', 'credit_score']
df_new['loan_eligible'] = credit_model.predict(df_new[rf_features])

for i, row in df_new.iterrows():
    print(f"Customer: {row['name']}, Credit Score: {row['credit_score']:.2f}, Loan Eligible: {row['loan_eligible']}")
    user_input = input(f"Is the prediction correct for {row['name']}? (y/n): ")
    if user_input.lower() == 'n':
        correct_label = int(input("Enter the correct loan eligibility (0/1): "))
    else:
        correct_label = int(row['loan_eligible'])

    row['gender'] = gender_encoder.inverse_transform([int(row['gender'])])[0]
    row['account_type'] = account_type_encoder.inverse_transform([int(row['account_type'])])[0]
    row['loan_history'] = loan_history_encoder.inverse_transform([int(row['loan_history'])])[0]

    new_entry = {
        'customer_id': row['customer_id'],
        'name': row['name'],
        'age': row['age'],
        'gender': row['gender'],
        'account_type': row['account_type'],
        'account_balance': row['account_balance'],
        'account_open_date': row['account_open_date'].strftime('%Y-%m-%d'),
        'loan_history': row['loan_history'],
        'transactions': row['transactions'],
        'loan_eligible': correct_label
    }

    pd.DataFrame([new_entry]).to_csv('training_data.csv', mode='a', header=False, index=False)
    print("✅ Added to training data.")

import credit_model  
