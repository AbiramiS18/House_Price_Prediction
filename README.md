# 🏠 House Price Prediction Model

A machine learning project that predicts house prices using regression models. This project includes data preprocessing, model training with multiple algorithms, and a Flask REST API for serving predictions.

## 🔍 Overview

This project implements a house price prediction system using various regression algorithms including:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The model is trained on housing data with features like square footage, number of bedrooms, bathrooms, location, and more.

## ✨ Features

- **Multiple ML Models**: Compare performance across Linear Regression, Random Forest, and XGBoost
- **Feature Engineering**: Includes derived features like house age, renovation status, and total square footage
- **Data Preprocessing**: Handles categorical variables with OneHotEncoder and uses sklearn pipelines
- **REST API**: Flask-based API for real-time predictions
- **Model Persistence**: Trained model saved using joblib for easy deployment

## 📁 Project Structure

```
Regression_Model/
├── app/
│   └── app.py              # Flask API application
├── data/
│   └── house_data.csv      # Training dataset
├── models/
│   └── house_price_model.pkl   # Trained model pipeline
├── train.py                # Model training script
├── requirements.text       # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Regression_Model.git
   cd Regression_Model
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.text
   ```

## 💻 Usage

### Training the Model

Run the training script to train all models and save the best one:

```bash
python train.py
```

This will:
- Load and preprocess the housing data
- Train Linear Regression, Random Forest, and XGBoost models
- Display evaluation metrics (RMSE, MAE, R²) for each model
- Save the Random Forest pipeline to `models/house_price_model.pkl`

### Running the API

Start the Flask development server:

```bash
python app/app.py
```

The API will be available at `http://127.0.0.1:5000`

### Making Predictions

Send a POST request to the `/predict` endpoint with house features:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1400,
    "sqft_basement": 400,
    "city": "Seattle",
    "statezip": "WA 98101",
    "house_age": 20,
    "renovated": 0,
    "total_sqft": 2200
  }'
```

**Response:**
```json
{
  "prediction": 450000.0,
  "status": "success"
}
```

## 🤖 Models

| Model | Description |
|-------|-------------|
| **Linear Regression** | Simple baseline model for comparison |
| **Random Forest** | Ensemble model with 100 decision trees |
| **XGBoost** | Gradient boosting with 500 estimators, learning rate 0.05, max depth 6 |

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Measures average prediction error
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and actual values
- **R²** (R-squared): Proportion of variance explained by the model

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check - returns API status |
| `/predict` | POST | Make a house price prediction |

## 🛠️ Technologies Used

- **Python 3.9.13**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting implementation
- **Flask** - REST API framework
- **Joblib** - Model serialization
- **Matplotlib & Seaborn** - Data visualization
