import pandas as pd 
import numpy as np
import pickle
import warnings

def impute_missing_values(df):

    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Education'] = df['Education'].fillna(df['Education'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].median())
    df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].median())
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Property_Area'] = df['Property_Area'].fillna(df['Property_Area'].mode()[0])

    return df

def encode_categorical(df):

    df['Gender'] = df['Gender'].apply(lambda x: 1 if x=='Male' else 0)
    df['Married'] = df['Married'].apply(lambda x: 1 if x=='Yes' else 0)
    df['Education'] = df['Education'].apply(lambda x: 1 if x == 'Graduate' else 0)
    df['Self_Employed'] = df['Self_Employed'].apply(lambda x: 1 if x == 'Yes' else 0)
    df['Credit_History'] = df['Credit_History'].astype(int)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype(str)

    # for one hot encoding, we may can use pd.get_dummies or scikit-learn
    # for this purpose, we wil do it manually since number of columns is small
    df['Dependents_0'] = df['Dependents'].apply(lambda x: 1 if x=='0' else 0)
    df['Dependents_1'] = df['Dependents'].apply(lambda x: 1 if x=='1' else 0)
    df['Dependents_2'] = df['Dependents'].apply(lambda x: 1 if x=='2' else 0)
    df['Dependents_3+'] = df['Dependents'].apply(lambda x: 1 if x=='3+' else 0)

    df['Loan_Amount_Term_12.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='12.0' else 0)
    df['Loan_Amount_Term_120.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='120.0' else 0)
    df['Loan_Amount_Term_180.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='180.0' else 0)
    df['Loan_Amount_Term_240.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='240.0' else 0)
    df['Loan_Amount_Term_300.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='300.0' else 0)
    df['Loan_Amount_Term_36.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='36.0' else 0)
    df['Loan_Amount_Term_360.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='360.0' else 0)
    df['Loan_Amount_Term_480.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='480.0' else 0)
    df['Loan_Amount_Term_60.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='60.0' else 0)
    df['Loan_Amount_Term_84.0'] = df['Loan_Amount_Term'].apply(lambda x:1 if x=='84.0' else 0)
    
    df['Property_Area_Rural'] = df['Property_Area'].apply(lambda x:1 if x=='Rural' else 0)
    df['Property_Area_Semiurban'] = df['Property_Area'].apply(lambda x:1 if x=='Semiurban' else 0)
    df['Property_Area_Urban'] = df['Property_Area'].apply(lambda x:1 if x=='Urban' else 0)

    df = df.drop(columns=['Dependents','Loan_Amount_Term','Property_Area'], axis=1)

    df = df[['Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History',
             'Dependents_0', 'Dependents_1', 'Dependents_2','Dependents_3+', 
             'Loan_Amount_Term_12.0', 'Loan_Amount_Term_120.0','Loan_Amount_Term_180.0', 
             'Loan_Amount_Term_240.0','Loan_Amount_Term_300.0', 'Loan_Amount_Term_36.0',
             'Loan_Amount_Term_360.0', 'Loan_Amount_Term_480.0','Loan_Amount_Term_60.0', 
             'Loan_Amount_Term_84.0', 'Property_Area_Rural','Property_Area_Semiurban',
             'Property_Area_Urban']]
    
    return df




# 'Loan_Amount_Term_12.0',    0
# 'Loan_Amount_Term_120.0',   4
# 'Loan_Amount_Term_180.0',   5
# 'Loan_Amount_Term_240.0',   6
# 'Loan_Amount_Term_300.0',   7
# 'Loan_Amount_Term_36.0',    1
# 'Loan_Amount_Term_360.0',   8
# 'Loan_Amount_Term_480.0',   9
# 'Loan_Amount_Term_60.0',    2
# 'Loan_Amount_Term_84.0'     3