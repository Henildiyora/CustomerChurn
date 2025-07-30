import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import boto3
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def upload_to_s3(file_path, file_name, bucket_name=None):
    """
    Upload a file to S3.
    
    Args:
        file_path (str): Local path to the file
        file_name (str): Name for the file in S3
        bucket_name (str): S3 bucket name (from .env if None)
    """
    try:
        bucket_name = bucket_name or os.getenv('S3_BUCKET')
        if not bucket_name:
            raise ValueError("S3_BUCKET not set in .env")
        
        s3 = boto3.client('s3')
        s3.upload_file(file_path, bucket_name, f'churn-data/{file_name}.csv')
        logger.info(f"Uploaded {file_name}.csv to s3://{bucket_name}/churn-data/")
    except Exception as e:
        logger.error(f"Failed to upload {file_name} to S3: {str(e)}")
        raise

def preprocessing(csv_path='data/telco_churn.csv'):
    """
    Preprocess the Telco dataset and upload train/validation CSVs to S3.
    
    Args:
        csv_path (str): Path to input CSV
    
    Returns:
        tuple: Paths to train, validation, and test CSVs
    """
    try:
        logger.info("Starting data preprocessing...")
        data = pd.read_csv(csv_path)

        # Handle missing values
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

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

        # Save splits to CSV
        os.makedirs('data/processed', exist_ok=True)
        train_data = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
        val_data = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1)
        test_data = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

        train_path = 'data/processed/train.csv'
        val_path = 'data/processed/validation.csv'
        test_path = 'data/processed/test.csv'

        train_data.to_csv(train_path, index=False, header=False)
        val_data.to_csv(val_path, index=False, header=False)
        test_data.to_csv(test_path, index=False, header=False)
        logger.info("Saved processed CSVs locally")

        # Upload to S3
        upload_to_s3(train_path, 'train')
        upload_to_s3(val_path, 'validation')
        logger.info("Preprocessing completed")

        return train_path, val_path, test_path

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise



