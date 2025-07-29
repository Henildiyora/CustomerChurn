import sagemaker
from dotenv import load_dotenv
import os
load_dotenv()


# Set up SageMaker session
sagemaker_session = sagemaker.Session()
role = os.getenv('AWS_ROLE_ARN')

# Reference the trained model
xgb = sagemaker.estimator.Estimator.attach('xgboost-2025-07-22-10-59-00-123')

# Deploy to an endpoint
predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',  
    serializer=sagemaker.serializers.CSVSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer()
)

# Print endpoint name
print(f"Endpoint name: {predictor.endpoint_name}")