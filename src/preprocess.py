import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import boto3
import os
from dotenv import load_dotenv
load_dotenv()

def upload_to_s3(file_path,file_name,bucket_name = 'telco-churn-s3'):

    s3 = boto3.client('s3')
    s3.upload_file(file_path,bucket_name,f'churn-data/{file_name}.csv')

def preprocessing(csv_path = 'data/telco_churn.csv'):

    data = pd.read_csv(csv_path)

    # Handle missing values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].mean())

    # Drop irrelevant columns
    data = data.drop(columns=['customerID'])

    # Encode categorical variables
    le = LabelEncoder() 
    for column in data.select_dtypes(include=['object']).columns: 
        if column != 'Churn': 
            data[column] = le.fit_transform(data[column])

    data['Churn'] = le.fit_transform(data['Churn'])

    # Split features and target
    X = data.drop(columns=['Churn']) 
    y = data['Churn']

    # Split into train (70%), validation (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) 
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save splits to CSV (SageMaker expects CSV without headers, target in first column)
    train_data = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1) 
    val_data = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1) 
    test_data = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

    os.makedirs('data/processed', exist_ok=True)
    train_data.to_csv('data/processed/train.csv', index=False, header=False) 
    val_data.to_csv('data/processed/validation.csv', index=False, header=False) 
    test_data.to_csv('data/processed/test.csv', index=False, header=False)

    upload_to_s3(file_path='data/processed/train.csv',file_name='train')
    upload_to_s3(file_path='data/processed/validation.csv',file_name='validation')



preprocessing()



