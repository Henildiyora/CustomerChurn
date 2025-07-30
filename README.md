# Customer Churn Prediction Pipeline

This project builds a machine learning model to predict telecom customer churn using AWS SageMaker and XGBoost. It includes data preprocessing, model training, deployment, and testing, orchestrated via `main.py`.

## Dataset
- **Name**: Telco Customer Churn (IBM sample dataset)
- **Description**: Contains ~7,000 customers' data (demographics, account info, services) and churn status.
- **Source**: [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Located in `data/telco_churn.csv`. No sensitive PII; data is fictional.

## Project Structure
- `data/`: Dataset and processed train/validation/test CSVs.
- `notebooks/`: Jupyter notebook for feature importance analysis.
- `model/`: Feature importance plot and optional model artifacts.
- `src/`: Scripts for preprocessing, training, deployment, and testing.
- `main.py`: Orchestrates the pipeline.
- `.env.example`: Template for environment variables.
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
   - Copy `.env.example` to `.env` and fill in `S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
   - Alternatively, configure AWS CLI: `aws configure`.
   - Create a SageMaker execution role with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`.
3. **Run Pipeline**:
   ```bash
   python main.py --bucket your-bucket-name
   ```
   This runs preprocessing, training, deployment, and testing. Optionally, specify `--training-job-name` or `--endpoint-name` to skip earlier steps.
4. **Feature Importance**:
   - Run `notebooks/Churn_Prediction_Pipeline.ipynb` to analyze feature importance.
5. **Cleanup**:
   - Manually delete the endpoint in AWS Console (SageMaker > Endpoints) to avoid charges.
   - Monitor AWS Billing to stay within free tier limits.

## AWS Free Tier Usage
- Uses `ml.t2.medium` for training (250 hours/month free) and hosting (750 hours for 2 months).
- S3: 5GB free storage.
- Delete endpoints to avoid charges.

## Results
- Predicts churn probability for telecom customers.
- Feature importance highlights key drivers like tenure and contract type.