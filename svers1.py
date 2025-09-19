import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")

# Function to get the full list of S&P 500 tickers (silent fallback)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'}) or soup.find('table', {'class': 'wikitable sortable'}) or soup.find('table')
        if table is None:
            return []
        tickers = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if not cols:
                continue
            ticker = cols[0].text.strip()
            ticker = ticker.split('.')[0].split(' ')[0].strip()
            if ticker:
                tickers.append(ticker.replace('\n', ''))
        return tickers
    except:
        return []

# Function to fetch historical stock data from Yahoo Finance
@st.cache_data(ttl=60 * 60 * 6)  # cache for 6 hours
def fetch_stock_data(tickers, period="10y"):
    try:
        if not tickers:
            return pd.DataFrame()

        data = yf.download(tickers, period=period, progress=False, threads=True)
        if data.empty:
            return pd.DataFrame()

        # Handle single ticker (Series or DataFrame with no MultiIndex)
        if isinstance(data, pd.Series):
            return data.to_frame(name=tickers[0])

        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                adj = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                adj = data["Close"]
            else:
                adj = data.xs(data.columns.levels[0][0], axis=1, level=0)
        else:
            if "Adj Close" in data.columns:
                adj = data[["Adj Close"]]
                adj.columns = [tickers[0]]
            elif "Close" in data.columns:
                adj = data[["Close"]]
                adj.columns = [tickers[0]]
            else:
                adj = data

        adj.columns = [str(c) for c in adj.columns]
        return adj
    except:
        return pd.DataFrame()

# Function to calculate log returns
def calculate_log_returns(historical_prices):
    return np.log(historical_prices / historical_prices.shift(1)).dropna(how='all')

# Function to calculate covariance matrix
def calculate_covariance_matrix(log_returns):
    return log_returns.cov()

# Function to fetch macroeconomic data from FRED (robust parsing)
def fetch_macro_data_from_fred(api_key, series_id, observations_limit=None):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
        if observations_limit is not None:
            url += f"&limit={observations_limit}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if 'observations' not in data:
            return pd.DataFrame()
        observations = data['observations']
        rows = []
        for obs in observations:
            date = obs.get('date')
            val = obs.get('value', '')
            try:
                valf = float(val)
            except:
                continue
            rows.append({'date': date, 'value': valf})
        if not rows:
            return pd.DataFrame()
        macro_data = pd.DataFrame(rows)
        macro_data['date'] = pd.to_datetime(macro_data['date'])
        macro_data.set_index('date', inplace=True)
        return macro_data
    except:
        return pd.DataFrame()

# Function to adjust returns based on macro data
def adjust_returns_based_on_macro(mean_log_returns, macro_data):
    if macro_data is None or macro_data.empty:
        return mean_log_returns
    normalized_macro = (macro_data - macro_data.mean()) / macro_data.std()
    try:
        latest_macro_value = normalized_macro.iloc[-1].values[0]
    except Exception:
        return mean_log_returns
    factor = np.clip(1 + latest_macro_value, 0.5, 1.5)
    return mean_log_returns * factor

# Function to simulate stock prices with multivariate shocks
def simulate_stock_prices(historical_prices, num_simulations, num_days, macro_data=None):
    if historical_prices is None or historical_prices.empty:
        return {}
    log_returns = calculate_log_returns(historical_prices)
    if log_returns.empty:
        return {}
    mean_log_returns = log_returns.mean()
    cov_log_returns = calculate_covariance_matrix(log_returns)

    mean_log_returns = adjust_returns_based_on_macro(mean_log_returns, macro_data)

    symbols = list(historical_prices.columns)
    n_stocks = len(symbols)
    simulated_prices = {stock: np.zeros((num_days, num_simulations)) for stock in symbols}

    last_prices = historical_prices.iloc[-1]
    for stock in symbols:
        simulated_prices[stock][0, :] = last_prices[stock]

    mean_vec = mean_log_returns.reindex(symbols).fillna(0).values
    cov_mat = cov_log_returns.reindex(index=symbols, columns=symbols).fillna(0).values

    try:
        for t in range(1, num_days):
            random_shocks = np.random.multivariate_normal(mean_vec, cov_mat + 1e-10 * np.eye(n_stocks), size=num_simulations)
            for i, stock in enumerate(symbols):
                simulated_prices[stock][t, :] = simulated_prices[stock][t - 1, :] * np.exp(random_shocks[:, i])
    except:
        return {}
    return simulated_prices

# Analyze simulation results
def analyze_simulations(simulated_prices, current_prices, increase_percentage, decrease_percentage):
    boom_probabilities, bust_probabilities = {}, {}
    for stock, sims in simulated_prices.items():
        if sims.size == 0:
            boom_probabilities[stock] = np.nan
            bust_probabilities[stock] = np.nan
            continue
        final_prices = sims[-1]
        boom_threshold = current_prices[stock] * (1 + increase_percentage)
        bust_threshold = current_prices[stock] * (1 - decrease_percentage)
        boom_probabilities[stock] = np.mean(final_prices > boom_threshold)
        bust_probabilities[stock] = np.mean(final_prices < bust_threshold)
    return boom_probabilities, bust_probabilities

# Plot histograms safely
def plot_histograms(simulated_prices):
    for stock, sims in simulated_prices.items():
        final_prices = sims[-1]
        plt.figure(figsize=(10, 5))
        plt.hist(final_prices, bins=50, edgecolor='k', alpha=0.7)
        plt.axvline(np.mean(final_prices), color='r', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(np.median(final_prices), color='g', linestyle='dashed', linewidth=1, label='Median')
        plt.title(f'Histogram of Final Simulated Prices for {stock}')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

def main():
    st.title('Stock Price Monte Carlo Simulation')

    sp500_tickers = get_sp500_tickers()
    st.sidebar.header('Simulation Parameters')

    default_tickers = ['AAPL', 'NVDA']
    use_manual = st.sidebar.checkbox("Enter tickers manually", value=False)
    if use_manual or not sp500_tickers:
        manual_input = st.sidebar.text_input("Tickers (comma separated)", value="AAPL, NVDA")
        selected_tickers = [t.strip().upper() for t in manual_input.split(',') if t.strip()]
    else:
        selected_tickers = st.sidebar.multiselect(
            'Select Stock Tickers (from S&P 500)',
            sp500_tickers,
            default=[t for t in default_tickers if t in sp500_tickers][:2]
        )

    if len(selected_tickers) < 1:
        st.stop()

    num_simulations = st.sidebar.slider('Number of Simulations', 100, 20000, 2000, step=100)
    num_days = st.sidebar.slider('Number of Days', 5, 252, 30, step=1)
    increase_percentage = st.sidebar.slider('Boom Return Threshold (fraction)', 0.0, 2.0, 0.2, step=0.01)
    decrease_percentage = st.sidebar.slider('Bust Return Threshold (fraction)', 0.0, 2.0, 0.2, step=0.01)

    period = st.sidebar.selectbox("Historical period", ["1y", "2y", "5y", "10y"], index=3)

    historical_prices = fetch_stock_data(selected_tickers, period=period)
    if historical_prices is None or historical_prices.empty:
        st.stop()

    st.subheader('Historical Stock Prices (last 5 rows)')
    st.dataframe(historical_prices.tail())

    st.subheader('Macroeconomic Data (optional)')
    api_key = st.sidebar.text_input("FRED API Key (optional)", value="")
    series_id = st.sidebar.text_input("FRED Series ID", value="CPIAUCSL")
    macro_data = pd.DataFrame()
    if api_key and series_id:
        macro_data = fetch_macro_data_from_fred(api_key, series_id)
        if not macro_data.empty:
            st.write(macro_data.tail())

    simulated_prices = simulate_stock_prices(historical_prices[selected_tickers], num_simulations, num_days, macro_data)
    if not simulated_prices:
        st.stop()

    st.subheader('Simulated Price Paths (sample of simulations)')
    for stock in selected_tickers:
        plt.figure(figsize=(10, 4))
        sample_count = min(50, num_simulations)
        plt.plot(simulated_prices[stock][:, :sample_count])
        plt.title(f'Simulated Price Paths for {stock}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        st.pyplot(plt.gcf())
        plt.close()

    st.subheader('Histograms of Final Simulated Prices')
    plot_histograms(simulated_prices)

    current_prices = historical_prices.iloc[-1].reindex(selected_tickers)
    boom_probabilities, bust_probabilities = analyze_simulations(simulated_prices, current_prices, increase_percentage, decrease_percentage)

    st.subheader('Boom Probabilities')
    for stock, prob in boom_probabilities.items():
        st.write(f"{stock}: {prob:.2%}" if not np.isnan(prob) else f"{stock}: N/A")

    st.subheader('Bust Probabilities')
    for stock, prob in bust_probabilities.items():
        st.write(f"{stock}: {prob:.2%}" if not np.isnan(prob) else f"{stock}: N/A")

if __name__ == "__main__":
    main()
