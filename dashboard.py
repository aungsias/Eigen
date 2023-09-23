# Data manipulation and analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Custom modules and functions
import capstone.portfolio.optimize as opt
from capstone.portfolio.prune import prune_recommended_portfolios
from capstone.model_selection import overunder_error
from capstone.utils import read_file, get_sectors

# Machine learning and modeling tools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# Progress bar for loops
from tqdm.auto import tqdm

import streamlit as st

class Burray:

    def __init__(self, forecast=126, cv=2):
        self.recommended_sector = None
        self.recommended_stocks = None
        self.maximum_sharpe_portfolio = None
        self.minimum_variance_portfolio = None
        self.risk_parity_portfolio = None
        self.best_model = None
        self.top_sectors = None
        self.mean_predicted_returns = None

        self._forecast = forecast
        self._cv = cv
        self._tscv = TimeSeriesSplit(self._cv)
        self._pca = make_pipeline(StandardScaler(), PCA(n_components=.8, random_state=42))
        self._models = {
            'ElasticNet': make_pipeline(StandardScaler(), ElasticNet(alpha=1, l1_ratio=0.5, random_state=42)),
            'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1, gamma='auto')),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
            'GradientBoost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        }
    
    def run(self):
        self._get_data()
        self._pca_transform()
        self._get_best_model()
        self._recommend_sector()
        self._recommend_constituents()
        self._get_optimal_allocations()

    def _get_data(self):
        self._master_data = read_file("master_df", index_col="Date")
        self._log_returns = read_file("snp_log_returns", index_col="Date")
        self._stocks_by_sector =read_file("stocks_by_sector", index_col=0)
        self._sectors = get_sectors()
        self._y_all = self._master_data[self._sectors]
        self._X_all = self._master_data[
            self._master_data.columns[
                ~self._master_data.columns.isin(self._y_all.columns)
            ]
        ]
    
    def _pca_transform(self):
        self._X_pca = pd.DataFrame(
            self._pca.fit_transform(self._X_all), 
            index=self._X_all.index
        )
        self._X_pca.columns = [f"PC{i+1}" for i in self._X_pca.columns]

    def _get_best_model(self):
        self._X_pca_shifted = self._X_pca.shift(self._forecast).dropna()
        self._X_pca_recent = X = self._X_pca_shifted.iloc[-self._forecast*2:]
        self._y_recent = self._y_all.iloc[-self._forecast*2:]
        ouls = {model: pd.DataFrame(index=self._sectors, columns=["MEAN_OUL"]) for model in self._models.keys()}
        for sector in tqdm(self._sectors):
            y = self._y_recent[sector]
            for name, model in self._models.items():
                cv_oul = []
                for train_idx, test_idx in self._tscv.split(self._X_pca_recent):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    model.fit(X_train, y_train)
                    y_hat_val = model.predict(X_test)
                    cv_oul.append(overunder_error(y_test, y_hat_val))
                oul_mean = np.mean(cv_oul)
                ouls[name].loc[sector, "MEAN_OUL"] = oul_mean
        mean_ouls = pd.DataFrame({model: oul.mean() for model, oul in ouls.items()})
        self.best_model = mean_ouls.idxmin(axis=1)[0]
    
    def _recommend_sector(self):
        future_start = self._y_all.index.max() + pd.DateOffset(1)
        future_end = future_start + pd.DateOffset(self._forecast - 1)
        future_dates = pd.date_range(future_start, future_end)
        self._X_train = self._X_pca[-self._forecast*2:-self._forecast]
        self._y_train = self._y_recent[-self._forecast:]
        self._X_test = self._X_pca[-self._forecast:]
        self._models[self.best_model].fit(self._X_train, self._y_train)
        predicted_returns = pd.DataFrame(self._models[self.best_model].predict(self._X_test), columns=self._sectors, index=future_dates)
        self.top_sectors = predicted_returns.cumsum().iloc[-1].sort_values(ascending=False)[:5]
        self.mean_predicted_returns = predicted_returns.mean()
        self.recommended_sector = self.mean_predicted_returns.idxmax()

    def _recommend_constituents(self):
        available_stocks = self._stocks_by_sector[self._stocks_by_sector["GICS Sector"] == self.recommended_sector]["Symbol"].to_list()
        self.recommended_stocks = [stock for stock in available_stocks if stock in self._log_returns.columns]
    
    def _get_optimal_allocations(self):
        weights = np.array([1/len(self.recommended_stocks)] * len(self.recommended_stocks))
        recent_returns = self._log_returns[-self._forecast:][self.recommended_stocks]
        max_sharpe_weights = opt.max_sharpe_opt(weights, recent_returns)[0]
        min_var_weights = opt.min_var_opt(weights, recent_returns)[0]
        risk_parity_weights = opt.risk_parity_opt(weights, recent_returns)[0]
        maximum_sharpe_portfolio = pd.Series(max_sharpe_weights, index=self.recommended_stocks)
        minimum_variance_portfolio = pd.Series(min_var_weights, index=self.recommended_stocks)
        risk_parity_portfolio = pd.Series(risk_parity_weights, index=self.recommended_stocks)
        self.maximum_sharpe_portfolio, self.minimum_variance_portfolio, self.risk_parity_portfolio = \
            prune_recommended_portfolios(
                maximum_sharpe_portfolio, minimum_variance_portfolio, risk_parity_portfolio
            )


def simulate_portfolio(portfolio_choice, initial_balance, allocation):
    allocation.name = portfolio_choice
    allocation = allocation.sort_values(ascending=False)
    simulated = allocation * initial_balance
    simulated = simulated.apply(lambda x: f"${x:,.2f}")
    st.write(f"Simulating {portfolio_choice} with initial balance of ${initial_balance:,.2f}")
    st.write("Allocations:", simulated)

# Initialize session state
if 'burray' not in st.session_state:
    st.session_state.burray = None
    st.session_state.recommendation_complete = False
    st.session_state.simulation_complete = False
    st.session_state.max_sharpe = None  # New
    st.session_state.min_var = None  # New
    st.session_state.risk_parity = None  # New

st.title('Stock Recommender')

if st.button('Recommend Stocks'):
    with st.spinner('Running the model...'):
        st.session_state.burray = Burray()
        st.session_state.burray.run()
        # Store portfolios in session state
        st.session_state.max_sharpe = st.session_state.burray.maximum_sharpe_portfolio
        st.session_state.min_var = st.session_state.burray.minimum_variance_portfolio
        st.session_state.risk_parity = st.session_state.burray.risk_parity_portfolio

    st.session_state.recommendation_complete = True
    st.success('Recommendation complete')

if st.session_state.recommendation_complete:
    portfolio_choice = st.selectbox(
        'Choose a portfolio strategy:',
        ('Maximum Sharpe', 'Minimum Variance', 'Risk Parity')
    )
    initial_balance = st.number_input('Enter initial balance:', min_value=1, value=1000)
    if st.button('Simulate Allocation'):
        if portfolio_choice == 'Maximum Sharpe':
            allocation = st.session_state.max_sharpe  
        elif portfolio_choice == 'Minimum Variance':
            allocation = st.session_state.min_var 
        elif portfolio_choice == 'Risk Parity':
            allocation = st.session_state.risk_parity
        simulate_portfolio(portfolio_choice, initial_balance, allocation)

    # Display allocations only if simulation is complete
    

