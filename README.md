# Cryptocurrency Risk Management & Price Prediction System

A Python-based application that analyzes cryptocurrency market data, predicts future prices using machine learning, and evaluates financial risk through key risk indicators.

## Overview

This project combines machine learning, market analysis, and risk assessment to help users understand cryptocurrency price movements and potential financial risks. It includes an interactive Streamlit dashboard for presenting market insights and visualizations.

## Application Screenshots



## Key Features

- **Price Prediction:** Uses a Random Forest Regressor to predict future cryptocurrency prices.
- **Market Analysis:** Analyzes historical cryptocurrency prices and trading volume.
- **Technical Indicators:** Calculates Simple Moving Averages (SMA) and Relative Strength Index (RSI).
- **Risk Assessment:** Measures volatility, Sharpe Ratio, Maximum Drawdown, and Value at Risk (VaR).
- **News Sentiment Analysis:** Classifies cryptocurrency news sentiment as positive, negative, or neutral using keyword-based analysis.
- **Interactive Dashboard:** Uses Streamlit and Plotly to display market data and visualizations.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data processing and numerical operations |
| Scikit-learn | Machine learning and feature scaling |
| Random Forest Regressor | Cryptocurrency price prediction |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |

## Machine Learning Approach

The project uses a Random Forest Regressor to predict cryptocurrency prices based on historical market data.

**Features used:**
- Historical price and volume lag features
- Simple Moving Averages (SMA)
- Relative Strength Index (RSI)
- MinMaxScaler for feature scaling

**Model evaluation metrics:**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Risk Analysis

The application calculates important financial risk indicators:

- **Volatility:** Measures price fluctuations.
- **Sharpe Ratio:** Evaluates returns in relation to volatility.
- **Maximum Drawdown:** Measures the largest decline from a previous peak.
- **Value at Risk (VaR):** Estimates potential losses at a 95% confidence level.

## Project Structure

```text
crypto-risk/
│
├── main.py
├── data_retrieval.py
├── ml_models.py
├── visualizations.py
├── utils.py
├── assets/
├── .streamlit/
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone <https://github.com/syedamanals86-debug/crypto-risk-management-and-prediction.git>
```

2. Navigate to the project directory:

```bash
cd crypto-risk
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the environment (Windows):

```bash
.venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the application:

```bash
streamlit run main.py
```

## Planned Improvements

- Explore additional machine learning models for price prediction.
- Improve prediction performance through hyperparameter tuning.
- Enhance news sentiment analysis using Natural Language Processing (NLP).
- Add more advanced financial risk indicators.
- Improve dashboard functionality and user experience.

## Author

**Syeda Rana Manal**

GitHub: https://github.com/syedamanals86-debug