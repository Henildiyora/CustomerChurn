import sagemaker
import pandas as pd
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from dotenv import load_dotenv
import os
import logging
import boto3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def predict(endpoint_name, test_csv='data/processed/test.csv'):
    """
    Test predictions using the deployed SageMaker endpoint.
    
    Args:
        endpoint_name (str): Name of the SageMaker endpoint
        test_csv (str): Path to test CSV
    
    Returns:
        list: Predictions for test samples
    """
    try:
        region = os.getenv('AWS_REGION', 'us-east-1')
        try:
            sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))
            logger.info("Initialized SageMaker session")
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker session: {str(e)}")
            raise

        # Load test data
        test_data = pd.read_csv(test_csv, header=None)
        test_features = test_data.iloc[:, 1:].values  
        logger.info("Loaded test data")

        # Set up predictor
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer()
        )

        # Test predictions
        logger.info("Testing predictions...")
        sample = test_features[0].tolist()
        prediction = predictor.predict(sample)
        logger.info(f"Sample prediction: {prediction}")

        # Test multiple samples
        predictions = [predictor.predict(row.tolist()) for row in test_features[:10]]
        logger.info(f"First 10 predictions: {predictions}")

        return predictions

    except Exception as e:
        logger.error(f"Prediction test failed: {str(e)}")
        raise