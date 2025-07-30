import sagemaker
from sagemaker.inputs import TrainingInput
import boto3
from dotenv import load_dotenv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def train(bucket_name=None):
    """
    Train an XGBoost model on SageMaker.
    
    Args:
        bucket_name (str): S3 bucket name (from .env if None)
    
    Returns:
        sagemaker.estimator.Estimator: Trained SageMaker estimator
    """
    try:
        bucket_name = bucket_name or os.getenv('S3_BUCKET')
        if not bucket_name:
            raise ValueError("S3_BUCKET not set in .env")

        # Initialize SageMaker session
        region = boto3.Session().region_name
        
        try:
            sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
            role = os.getenv('AWS_ROLE_ARN')
            logger.info("Initialized SageMaker session and role")
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker session: {str(e)}")
            raise

        # Define S3 paths
        prefix = 'churn-data'
        s3_train_path = f's3://{bucket_name}/{prefix}/train.csv'
        s3_val_path = f's3://{bucket_name}/{prefix}/validation.csv'
        output_path = f's3://{bucket_name}/output'

        # Get XGBoost container
        container = sagemaker.image_uris.retrieve('xgboost', region, 'latest')
        logger.info("Retrieved XGBoost container URI")

        # Configure Estimator
        xgb = sagemaker.estimator.Estimator(
            container,
            role,
            instance_count=1,
            instance_type='ml.t2.medium',  
            output_path=output_path,
            sagemaker_session=sagemaker_session
        )

        # Set hyperparameters
        xgb.set_hyperparameters(
            objective='binary:logistic',
            num_round=100,
            max_depth=5,
            eta=0.2,
            subsample=0.8
        )

        # Define training inputs
        train_input = TrainingInput(s3_train_path, content_type='csv')
        val_input = TrainingInput(s3_val_path, content_type='csv')

        # Launch training job
        logger.info("Starting SageMaker training job...")
        xgb.fit({'train': train_input, 'validation': val_input})
        logger.info("Training job completed")

        return xgb

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise