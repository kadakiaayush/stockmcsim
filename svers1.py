import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

# Function to get the full list of S&P 500 tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        tickers.append(cols[0].text.strip())
    return tickers

# Function to fetch historical stock data from Yahoo Finance
@st.cache_data
def fetch_stock_data(tickers, period="10y"):
    try:
        st.write("Fetching stock data...")
        data = yf.download(tickers, period=period)
        if data.empty:
            st.error("No data fetched for the given tickers. Please check the tickers or try again later.")
            st.stop()
        return data['Adj Close'] if 'Adj Close' in data else data['Close']
    except Exception as e:
        st.error(f"An error occurred while fetching stock data: {e}")
        st.stop()

# Function to calculate log returns
def calculate_log_returns(historical_prices):
    return np.log(historical_prices / historical_prices.shift(1)).dropna()

# Function to calculate covariance matrix
def calculate_covariance_matrix(log_returns):
    return log_returns.cov()

# Function to fetch macroeconomic data from FRED
def fetch_macro_data_from_fred(api_key, series_id):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
        response = requests.get(url)
        data = response.json()
        if 'observations' not in data:
            st.error("No macroeconomic data fetched. Please check the API key and series ID.")
            st.stop()
        observations = data['observations']
        dates = [obs['date'] for obs in observations]
        values = [float(obs['value']) for obs in observations]
        macro_data = pd.DataFrame({'date': dates, 'value': values})
        macro_data['date'] = pd.to_datetime(macro_data['date'])
        macro_data.set_index('date', inplace=True)
        return macro_data
    except Exception as e:
        st.error(f"An error occurred while fetching macroeconomic data: {e}")
        st.stop()

# Function to adjust returns based on macro data
def adjust_returns_based_on_macro(mean_log_returns, macro_data):
    normalized_macro = (macro_data - macro_data.mean()) / macro_data.std()
    latest_macro_value = normalized_macro.iloc[-1].values[0]
    return mean_log_returns * (1 + latest_macro_value)

# Function to simulate stock prices
def simulate_stock_prices(historical_prices, num_simulations, num_days, macro_data):
    log_returns = calculate_log_returns(historical_prices)
    mean_log_returns = log_returns.mean()
    cov_log_returns = calculate_covariance_matrix(log_returns)
    mean_log_returns = adjust_returns_based_on_macro(mean_log_returns, macro_data)
    simulated_prices = {stock: np.zeros((num_days, num_simulations)) for stock in historical_prices.columns}
    for stock in historical_prices.columns:
        simulated_prices[stock][0] = historical_prices[stock].iloc[-1]
    for t in range(1, num_days):
        random_shocks = np.random.multivariate_normal(mean_log_returns, cov_log_returns, num_simulations)
        for i, stock in enumerate(historical_prices.columns):
            simulated_prices[stock][t] = simulated_prices[stock][t-1] * np.exp(random_shocks[:, i])
    return simulated_prices

# Analyze simulation results
def analyze_simulations(simulated_prices, current_prices, increase_percentage, decrease_percentage):
    boom_probabilities, bust_probabilities = {}, {}
    for stock, simulations in simulated_prices.items():
        boom_threshold = current_prices[stock] * (1 + increase_percentage)
        bust_threshold = current_prices[stock] * (1 - decrease_percentage)
        final_prices = simulations[-1]
        boom_probabilities[stock] = np.mean(final_prices > boom_threshold)
        bust_probabilities[stock] = np.mean(final_prices < bust_threshold)
    return boom_probabilities, bust_probabilities

# Plot histograms
def plot_histograms(simulated_prices):
    for stock, simulations in simulated_prices.items():
        final_prices = simulations[-1]
        plt.figure(figsize=(10, 6))
        plt.hist(final_prices, bins=50, edgecolor='k', alpha=0.7)
        plt.axvline(np.mean(final_prices), color='r', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(np.median(final_prices), color='g', linestyle='dashed', linewidth=1, label='Median')
        plt.title(f'Histogram of Final Simulated Prices for {stock}')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        st.pyplot(plt)

# Define the Streamlit app
def main():
    st.title('Stock Price Simulation and Analysis')

    st.markdown(
    """
    <div style="
        background-color: #e7f3fe;
        color: #31708f;
        border-left: 5px solid #b3d8fd;
        padding: 16px;
        margin-top: 10px;
        font-size: 16px;
        border-radius: 6px;">
        💡 <strong>Tip:</strong> Click the <b>menu arrow in the top-left corner</b> to open settings and customize simulation parameters like stock tickers, time horizon, and return thresholds.
    </div>
    """,
    unsafe_allow_html=True
)


    sp500_tickers = get_sp500_tickers()
    st.sidebar.header('Simulation Parameters')
    
    default_tickers = ['AAPL', 'NVDA']
    selected_tickers = st.sidebar.multiselect(
        'Select Stock Tickers',
        sp500_tickers,
        default=[ticker for ticker in default_tickers if ticker in sp500_tickers][:2]
    )

    if len(selected_tickers) < 2:
        st.warning("Please select at least two tickers to enable simulation with a covariance matrix.")
        st.stop()

    num_simulations = st.sidebar.slider('Number of Simulations', 1000, 10000, 1000, 1000)
    num_days = st.sidebar.slider('Number of Days', 10, 50, 30, 5)
    increase_percentage = st.sidebar.slider('Boom Return Threshold', 0.0, 0.5, 0.2, 0.05)
    decrease_percentage = st.sidebar.slider('Bust Return Threshold', 0.0, 0.5, 0.2, 0.05)

    historical_prices = fetch_stock_data(selected_tickers)
    st.subheader('Historical Stock Prices')
    st.write(historical_prices.tail())

    st.subheader('Macroeconomic Data')
    api_key = "08828fc4fc9dbcfbea6f77718987ade3"
    series_id = "CPIAUCSL"
    macro_data = fetch_macro_data_from_fred(api_key, series_id)
    st.write(macro_data.tail())

    simulated_prices = simulate_stock_prices(historical_prices, num_simulations, num_days, macro_data)
    st.subheader('Simulated Price Paths')
    for stock in selected_tickers:
        plt.figure(figsize=(10, 6))
        plt.plot(simulated_prices[stock])
        plt.title(f'Simulated Price Paths for {stock}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        st.pyplot(plt)

    st.subheader('Histograms of Final Simulated Prices')
    plot_histograms(simulated_prices)

    current_prices = historical_prices.iloc[-1]
    boom_probabilities, bust_probabilities = analyze_simulations(simulated_prices, current_prices, increase_percentage, decrease_percentage)

    st.subheader('Boom Probabilities')
    for stock, prob in boom_probabilities.items():
        st.write(f"{stock}: {prob:.2%}")

    st.subheader('Bust Probabilities')
    for stock, prob in bust_probabilities.items():
        st.write(f"{stock}: {prob:.2%}")

if __name__ == "__main__":
    main()
