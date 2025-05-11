import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('training_data.csv')

# Convert date and extract account age
df['account_open_date'] = pd.to_datetime(df['account_open_date'])
df['account_years'] = datetime.now().year - df['account_open_date'].dt.year

# Extract transaction features
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

df[['deposit_total', 'withdrawal_total', 'num_transactions', 'avg_transaction']] = df['transactions'].apply(extract_transaction_features)

# Encode categorical features
gender_encoder = LabelEncoder()
account_type_encoder = LabelEncoder()
loan_history_encoder = LabelEncoder()

df['gender'] = gender_encoder.fit_transform(df['gender'])
df['account_type'] = account_type_encoder.fit_transform(df['account_type'])
df['loan_history'] = loan_history_encoder.fit_transform(df['loan_history'])

# Save encoders
joblib.dump(gender_encoder, 'gender_encoder.pkl')
joblib.dump(account_type_encoder, 'account_type_encoder.pkl')
joblib.dump(loan_history_encoder, 'loan_history_encoder.pkl')

# Train credit score model (Linear Regression)
credit_features = ['account_balance', 'account_years', 'loan_history', 'account_type', 'num_transactions', 'avg_transaction']
credit_model = LinearRegression()
df['credit_score'] = credit_model.fit(df[credit_features], df['loan_eligible']).predict(df[credit_features])

joblib.dump(credit_model, 'credit_score_model.pkl')

# Train loan eligibility model (Random Forest)
rf_features = credit_features + ['gender', 'credit_score']
loan_model = RandomForestClassifier(n_estimators=100, random_state=42)
loan_model.fit(df[rf_features], df['loan_eligible'])

# Save loan model
joblib.dump(loan_model, 'credit_model.pkl')

print("✅ Models trained and saved.")
