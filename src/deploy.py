import sagemaker
from sagemaker import get_execution_role
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

def deploy(training_job_name):
    """
    Deploy a trained SageMaker model to an endpoint.
    
    Args:
        training_job_name (str): Name of the SageMaker training job
    
    Returns:
        sagemaker.predictor.Predictor: Deployed predictor
    """
    try:
        # Attach to the trained model
        xgb = sagemaker.estimator.Estimator.attach(training_job_name)
        logger.info(f"Attached to training job: {training_job_name}")

        # Deploy to an endpoint
        predictor = xgb.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',  
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer()
        )
        logger.info(f"Model deployed to endpoint: {predictor.endpoint_name}")

        return predictor

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise