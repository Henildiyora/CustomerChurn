import sagemaker
import boto3 
import os
from dotenv import load_dotenv
load_dotenv()

# Set up SageMaker session
sagemaker_session = sagemaker.Session()
region = boto3.Session().region_name
role = os.getenv('AWS_ROLE_ARN')

# define s3 paths 
bucket = 'telco-churn-s3'
prefix = 'churn-data'
s3_train_path = f's3://{bucket}/{prefix}/train.csv'
s3_val_path = f's3://{bucket}/{prefix}/validation.csv'
output_path = f's3://{bucket}/output'

# Get Xgboost container
container = sagemaker.image_uris.retrieve('xgboost',region,'latest')

# Configure the Estimeter 
xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count = 1,
    instance_type = 'ml.m5.large',
    output_path=output_path,
    sagemaker_session = sagemaker_session
)

# Set hyperparameters
xgb.set_hyperparameters(
    objective='binary:logistic',
    num_round=100,
    max_depth=5,
    eta=0.2,
    subsample=0.8
)

# Define input channels
train_input = sagemaker.inputs.TrainingInput(s3_train_path,content_type = 'csv')
val_input = sagemaker.inputs.TrainingInput(s3_val_path,content_type = 'csv')

# Launch training job
xgb.fit({'train': train_input, 'validation': val_input})