import sagemaker
import pandas as pd
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Load test data
test_data = pd.read_csv('data/processed/test.csv', header=None)
test_features = test_data.iloc[:, 1:].values  # Exclude target column

# Set up predictor
predictor = Predictor(
    endpoint_name=' ',  # Replace with your endpoint name
    sagemaker_session=sagemaker.Session(),
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

# Test prediction
sample = test_features[0].tolist()  # Take one sample
prediction = predictor.predict(sample)
print(f"Prediction: {prediction}")

# Evaluate multiple samples
predictions = []
for row in test_features[:10]:  # Test first 10 samples
    pred = predictor.predict(row.tolist())
    predictions.append(pred)
print(f"First 10 predictions: {predictions}")