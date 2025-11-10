<<<<<<< HEAD
# california-housing-ml
Machine Learning project for predicting California house prices using Scikit-learn. Includes preprocessing pipelines, model comparison, and production-ready inference.
=======
# California Housing Price Prediction

A complete machine learning system for predicting median house values in California using the California Housing dataset. Features production-ready code with separate training and inference pipelines, model persistence, and comprehensive evaluation.

## Features
- **Stratified Train-Test Split** for balanced data distribution
- **Automated ML Pipeline** with preprocessing and feature engineering
- **Multiple Model Comparison**: Linear Regression, Decision Tree, Random Forest
- **10-Fold Cross-Validation** for robust model evaluation
- **Production-Ready Inference** with model and pipeline persistence
- **Batch Prediction** capability for new data

## Tech Stack
- **Python**: Core programming language
- **Scikit-learn**: ML algorithms, preprocessing, and pipelines
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Joblib**: Model serialization and persistence

## Models Evaluated
1. **Linear Regression**: Baseline model
2. **Decision Tree Regressor**: Non-linear relationships
3. **Random Forest Regressor**: Best performer (selected for deployment)

## Data Preprocessing Pipeline
- **Numerical Features**: Median imputation → Standard scaling
- **Categorical Features**: One-hot encoding with unknown category handling
- **Feature Engineering**: Income category stratification for balanced splits

## Project Structure
```
├── housing.csv                 # California housing dataset
├── train_model.py             # Training script with model comparison
├── deploy_model.py            # Production-ready train/inference script
├── model.pkl                  # Saved Random Forest model
├── pipeline.pkl               # Saved preprocessing pipeline
├── input_test.csv             # Sample input for predictions
├── output.csv                 # Predictions output
└── README.md                  # Project documentation
```

## Usage

### Training Phase (First Run)
```bash
python deploy_model.py
```
- Trains Random Forest model on housing dataset
- Saves trained model and preprocessing pipeline as `.pkl` files
- Displays: "Model trained and saved."

### Inference Phase (Subsequent Runs)
```bash
python deploy_model.py
```
- Automatically detects existing model
- Loads saved model and pipeline
- Reads `input_test.csv` for new predictions
- Outputs results to `output.csv`

### Model Comparison & Evaluation
```bash
python train_model.py
```
Outputs:
- Training RMSE for all models
- 10-fold Cross-Validation RMSE scores
- Statistical summary (mean, std, min, max)

## Model Performance
| Model | Training RMSE | CV Mean RMSE |
|-------|---------------|--------------|
| Linear Regression | ~68,000 | ~69,000 |
| Decision Tree | 0 (overfit) | ~70,000 |
| **Random Forest** | **~18,000** | **~50,000** ✓ |

*Random Forest selected for production due to best generalization*

## Input Data Format
Your `input_test.csv` should contain these columns:
```
longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
population, households, median_income, ocean_proximity
```

## Key Implementation Details
- **Stratified Splitting**: Income-based stratification ensures representative test sets
- **Pipeline Architecture**: Automated preprocessing prevents data leakage
- **Model Persistence**: Joblib serialization for efficient model storage
- **Error Handling**: Pipeline handles missing values and unknown categories

## Future Enhancements
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Web API for real-time predictions
- Docker containerization for deployment
>>>>>>> a5acccb (Initial commit - California Housing Price Prediction project)
