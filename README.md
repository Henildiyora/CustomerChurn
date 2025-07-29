# Customer Churn Prediction Pipeline

This project builds a machine learning model to predict telecom customer churn using AWS SageMaker and XGBoost. It includes data preprocessing, model training on AWS, deployment of a real-time inference endpoint, and analysis of feature importance for insights into customer attrition.

## Dataset
- **Name**: Telco Customer Churn (IBM sample dataset)
- **Description**: Contains information about ~7,000 customers (demographics, account info, services) and whether they churned.
- **Source**: [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- The raw dataset is in `data/telco_churn.csv`. No sensitive PII; data is fictional.

## Project Structure
- `data/`: Dataset and processed train/validation/test CSVs.
- `src/`: Scripts for preprocessing, training, deployment, and inference.
- `notebooks/`: Jupyter notebook for feature importance analysis.
- `model/`: Feature importance plot and optional model artifacts.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Setup Instructions
1. **Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   Note: Developed on macOS M1; dependencies are compatible with Apple Silicon.
2. **AWS Credentials**:
   - Configure AWS CLI: `aws configure`
   - Create a SageMaker execution role with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`.
3. **Data Preparation**:
   ```bash
   python src/preprocess.py
   ```
   Outputs `train.csv`, `validation.csv`, `test.csv` in `data/processed/` and uploads to S3.
4. **Model Training**:
   ```bash
   python src/train_sagemaker.py
   ```
   Launches a SageMaker training job using XGBoost.
5. **Deploy Model**:
   ```bash
   python src/deploy.py
   ```
   Deploys the model to a SageMaker endpoint. Record the endpoint name.
6. **Test Inference**:
   ```bash
   python src/predict_test.py --endpoint your-endpoint-name
   ```
   Sends sample data to the endpoint and prints predictions.
7. **Feature Importance**:
   - Run `notebooks/Churn_Prediction_Pipeline.ipynb` to generate a feature importance plot.
8. **Cleanup**:
   - Delete the endpoint: `predictor.delete_endpoint()`
   - Stop any running SageMaker notebook instances.
   - Monitor AWS Billing to stay within free tier limits.

## AWS Free Tier Usage
- Uses `ml.t2.medium` for training (250 hours/month free) and hosting (750 hours for 2 months).
- S3: 5GB free storage.
- Delete endpoints and stop instances to avoid charges.

## Results
- The model predicts churn probability for telecom customers.
- Feature importance analysis highlights key drivers like tenure and contract type.