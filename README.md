# Credit Card Fraud Detection API

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using an ensemble classification model, served through FastAPI and containerized with Docker.

## Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent a very small proportion of total transactions.

This project builds a machine learning pipeline that classifies anonymized credit card transactions as:

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

The trained model is deployed as a REST API, allowing external applications to submit transaction features and receive fraud predictions and probability scores.

## Key Features

- Exploratory Data Analysis (EDA)
- Handling highly imbalanced datasets
- Feature selection
- Hybrid resampling using SMOTE and Random Undersampling
- Feature scaling using MinMaxScaler
- Ensemble classification using VotingClassifier
- Threshold-based classification
- REST API development using FastAPI
- Request validation using Pydantic
- Docker containerization
- Health check endpoint

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, CatBoost |
| Imbalanced Data | SMOTE, Random Undersampling |
| Visualization | Matplotlib, Seaborn |
| API Development | FastAPI, Pydantic |
| API Testing | Swagger UI, Python Requests |
| Deployment | Docker |
| Development | Jupyter Notebook, VS Code |

## Machine Learning Pipeline

The project follows these main stages:

1. **Data Exploration:** Analyze transaction distributions, missing values, class imbalance, and feature patterns.
2. **Feature Selection:** Select the following 10 features based on the notebook analysis:

   `V17, V14, V12, V10, V16, V7, V3, V11, V4, V18`

3. **Data Balancing:** Apply SMOTE followed by Random Undersampling to address class imbalance.
4. **Feature Scaling:** Use MinMaxScaler, fitted on the original training data.
5. **Model Training:** Train an ensemble VotingClassifier.
6. **Threshold Optimization:** Use a classification threshold of approximately `0.74455`.
7. **Model Evaluation:** Evaluate using metrics suitable for imbalanced classification, including Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

## API Endpoints

The trained model and preprocessing scaler are loaded from `model.pkl`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/predict` | Predict whether a transaction is fraudulent |

### Example Prediction Request

```json
{
  "V17": -1,
  "V14": -1,
  "V12": -1,
  "V10": -1,
  "V16": -1,
  "V7": -1,
  "V3": -1,
  "V11": -1,
  "V4": -1,
  "V18": -1
}
```

### Example Response

```json
{
  "prediction": 0,
  "result": "Legitimate",
  "fraud_probability": 0.0667,
  "threshold": 0.7445532313170999
}
```

The response includes:

- `prediction`: Predicted class (0 or 1).
- `result`: Human-readable classification.
- `fraud_probability`: Estimated probability of fraud.
- `threshold`: Decision threshold used for classification.

## Running with Docker

### 1. Clone the repository

```bash
git clone https://github.com/MahmoudNashat950/credit_fraud_project.git

cd credit_fraud_project
```

### 2. Build the Docker image

```bash
docker build -t credit-fraud-api:v1 .
```

### 3. Run the container

```bash
docker run -d \
  --name credit-fraud-container \
  -p 8000:8000 \
  credit-fraud-api:v1
```

### 4. Check API health

Open:

`http://localhost:8000/health`

Expected response:

```json
{
  "status": "healthy"
}
```

### 5. Access API documentation

Open:

`http://localhost:8000/docs`

Use Swagger UI to test the prediction endpoint.

## Testing

The API has been tested with:

- Successful health check requests (`200 OK`).
- Valid prediction requests (`200 OK`).
- Missing required features (`422 Unprocessable Entity`).
- Invalid feature data types (`422 Unprocessable Entity`).
- External Python HTTP requests to the Dockerized API.

## Project Structure

```text
credit_fraud_project/
│
├── app.py
├── credit_fraud_train.py
├── credit_fraud_utils_data.py
├── credit_fraud_utils_eval.py
├── fraud_exploration.ipynb
├── model.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Future Improvements

- Add automated unit and integration tests.
- Introduce CI/CD pipelines.
- Deploy the API to a cloud platform.
- Add monitoring and logging.
- Implement model performance and data drift monitoring.
- Add batch prediction support.
- Improve reproducibility of model training and evaluation.

## Limitations

- The model uses anonymized transaction features.
- Predictions depend on the selected feature set and the training distribution.
- The current API is a model-serving prototype, not a production banking fraud prevention system.
- Real-world performance should be validated using an independent test dataset.

## Author

**Mahmoud Nashaat**

Software Engineering Student | Machine Learning & Data Engineering

GitHub: [MahmoudNashat950](https://github.com/MahmoudNashat950)