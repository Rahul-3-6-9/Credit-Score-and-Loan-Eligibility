## Project Overview
This project is an **ATM Management System** built in Python, which incorporates a **machine learning-based credit scoring system**. The system handles customer transactions, performs credit score calculations, and predicts loan eligibility using **Linear Regression** for credit scoring and **Random Forest** for loan eligibility prediction.

## Features
- **Customer Transactions Management**: Deposit, withdrawal, and transaction history tracking.
- **Credit Scoring**: Calculates credit score using a **Linear Regression** model based on customer transactions, loan history, account type, account balance, and account age.
- **Loan Eligibility Prediction**: Uses a **Random Forest** classifier to predict loan eligibility based on the calculated credit score and other features.
- **Dynamic Data Handling**: The system can update its credit scoring model with real-time transaction data and feedback from the user.

## Project Structure
ATM-Management-System/
│
├── data/
│ ├── training_customers.csv # Customer training dataset
│ └── test_customers.csv # Customer test dataset
│
├── main.py # Main ATM system logic
├── credit_model.py # Machine learning model for credit scoring
├── process_uploaded_customers.py # Script for processing customer data and updating models
├── setup_db.py # Script for setting up the SQLite database
└── README.md # Project documentation

## Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `sklearn`, `joblib`, `sqlite3`, `matplotlib`

## Installation
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/atm-management-system.git
    cd atm-management-system
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Running the ATM Management System
The system can be run by executing the `main.py` file:

```bash
  python main.py
```

This will start the ATM system, where you can interact with the available functionalities.

### 2. Training the Credit Scoring Model
To train or retrain the credit scoring model, run the credit_model.py script:

```bash
  python credit_model.py
```

This will train the Linear Regression model for credit scoring based on the customer data.

### 3. Processing New Customer Data
The process_uploaded_customers.py script allows you to process new customer data, update the training dataset, and retrain the model:

```bash
python process_uploaded_customers.py
```

### 4. Setup Database
To set up the SQLite database for storing transaction data, run the setup_db.py script:

```bash
python setup_db.py
```
