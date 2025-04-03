# Harveston Climate Forecasting

**DataCrunch 2025 Competition**  
**University of Moratuwa**

## Project Overview

This repository contains our solution for the DataCrunch 2025 competition, focused on predicting critical environmental variables for Harveston, a self-sufficient agricultural region experiencing climate shifts. Our models forecast five key climate variables to help farmers make informed decisions about planting cycles, resource allocation, and preparation for weather extremes.

### Forecasting Targets

1. Average Temperature (°C)
2. Radiation (W/m²)
3. Rain Amount (mm)
4. Wind Speed (km/h)
5. Wind Direction (°)

## Repository Structure

```
/Data_Crunch_001
├── data/                           
│   ├── sample_submission.csv       
│   ├── test.csv                    
│   └── train.csv                   
│
├── Notebooks_and_Scripts/
│   ├── 01_EDA_and_Analysis.ipynb   # Exploratory analysis
│   ├── 02_model_training.ipynb     # Model development
│   ├── utils.py
│   └── plots/                      # Generated visualizations
├── utils.py
│
├── final_submission.csv            
├── technical_report.pdf            
├── README.md                      
└── requirements.txt                                
```

## Environment Setup

### Requirements

The required libraries for this project are listed in `requirements.txt`. The main dependencies include:

- numpy
- pandas
- scikit-learn
- lightgbm
- optuna
- matplotlib
- seaborn

### Installation

1. Clone this repository:
```
git clone [https://github.com/your-github/Data_Crunch_001.git](https://github.com/FouetteBytes/Data_Crunch_106.git)
cd Data_Crunch_106
```

2. Create and activate a virtual environment (optional):
```
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install required packages:
```
pip install -r requirements.txt
```

## Solution Approach

Our solution employs a comprehensive approach to time series forecasting, combining advanced feature engineering with ensemble modeling techniques:

### Data Preprocessing
- Temperature unit standardization (Kelvin to Celsius)
- Missing value imputation using hierarchical methods
- Outlier detection and treatment
- Geographic clustering of kingdoms

### Feature Engineering
- Temporal features with cyclical encoding
- Lagged variables (1-30 days)
- Rolling window statistics (multiple window sizes)
- Exponentially weighted moving averages
- Differencing features for trend removal
- Cross-feature interactions

### Modeling
- Primary model: LightGBM with optimized hyperparameters
- Multi-seed ensemble approach (3 seeds)
- Target-specific feature selection
- Time-series cross-validation
- Specialized handling for directional data (Wind Direction)

## Running the Code

### Exploratory Data Analysis

```
jupyter notebook 01_EDA_and_Analysis.ipynb
```

This notebook performs comprehensive data exploration, visualizing distributions, temporal patterns, and correlations between variables.

### Model Training and Prediction

```
jupyter notebook 02_model_training.ipynb
```

This notebook implements the complete modeling pipeline:
1. Data preprocessing
2. Feature engineering
3. Model training with hyperparameter optimization
4. Ensemble prediction
5. Submission file generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
