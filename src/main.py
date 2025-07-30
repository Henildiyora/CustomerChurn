import argparse
from dotenv import load_dotenv
import logging
from preprocess import preprocessing, upload_to_s3
from train_sagemaker import train
from deploy import deploy
from predict_test import predict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument('--bucket', help='S3 bucket name (overrides .env)')
    parser.add_argument('--input-csv', default='data/telco_churn.csv', help='Input CSV path')
    parser.add_argument('--training-job-name', help='SageMaker training job name for deployment')
    parser.add_argument('--endpoint-name', help='SageMaker endpoint name for predictions')
    args = parser.parse_args()

    try:
        # Step 1: Preprocess data
        logger.info("Starting pipeline...")
        train_path, val_path, test_path = preprocessing(args.input_csv)
        logger.info("Preprocessing completed")

        # Step 2: Train model
        training_job = train(args.bucket)
        logger.info(f"Training job completed: {training_job.latest_training_job.name}")

        # Step 3: Deploy model
        predictor = deploy(training_job.latest_training_job.name)
        logger.info(f"Deployed endpoint: {predictor.endpoint_name}")

        # Step 4: Test predictions
        predictions = predict(predictor.endpoint_name, test_path)
        logger.info("Prediction testing completed")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()