import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup

st.set_page_config(layout="wide")

# Function to get the full list of S&P 500 tickers (robust)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Preferentially find by known id, otherwise look for first wikitable sortable
        table = soup.find('table', {'id': 'constituents'})
        if table is None:
            table = soup.find('table', {'class': 'wikitable sortable'})
        if table is None:
            # Final fallback: first table on the page
            table = soup.find('table')
        if table is None:
            st.error("Could not find the S&P 500 constituents table on Wikipedia.")
            return []
        tickers = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if not cols:
                continue
            # Usually ticker is in first column, but sanitize
            ticker = cols[0].text.strip()
            # Tickers on Wikipedia sometimes include footnote marks — keep only valid chars
            ticker = ticker.split('.')[0].split(' ')[0].strip()
            if ticker:
                tickers.append(ticker.replace('\n', ''))
        return tickers
    except Exception as e:
        st.warning(f"Failed to fetch S&P 500 tickers from Wikipedia: {e}")
        return []

# Function to fetch historical stock data from Yahoo Finance
@st.cache_data(ttl=60 * 60 * 6)  # cache for 6 hours
@st.cache_data(ttl=60 * 60 * 6)  # cache for 6 hours
def fetch_stock_data(tickers, period="10y"):
    try:
        if not tickers:
            st.error("No tickers provided.")
            return pd.DataFrame()
        st.info("Fetching stock data from Yahoo Finance...")

        data = yf.download(tickers, period=period, progress=False, threads=True)
        if data.empty:
            st.error("No data fetched for the given tickers. Please check the tickers or try again later.")
            return pd.DataFrame()

        # Handle single ticker (Series or DataFrame with no MultiIndex)
        if isinstance(data, pd.Series):
            return data.to_frame(name=tickers[0])

        if isinstance(data.columns, pd.MultiIndex):
            # yfinance multi-ticker DataFrame
            if "Adj Close" in data.columns.get_level_values(0):
                adj = data["Adj Close"]
            elif "Close" in data.columns.get_level_values(0):
                adj = data["Close"]
            else:
                adj = data.xs(data.columns.levels[0][0], axis=1, level=0)
        else:
            # Single ticker DataFrame with simple columns
            if "Adj Close" in data.columns:
                adj = data[["Adj Close"]]
                adj.columns = [tickers[0]]
            elif "Close" in data.columns:
                adj = data[["Close"]]
                adj.columns = [tickers[0]]
            else:
                adj = data

        # Make sure column names are clean ticker symbols
        adj.columns = [str(c) for c in adj.columns]
        return adj
    except Exception as e:
        st.error(f"An error occurred while fetching stock data: {e}")
        return pd.DataFrame()


        # Prefer 'Adj Close' if available (multi-ticker yields a DataFrame with columns like ('Adj Close', 'AAPL'))
        if isinstance(data, pd.DataFrame) and ('Adj Close' in data.columns or ('Adj Close' in data.columns.levels if hasattr(data.columns, 'levels') else False)):
            # If MultiIndex columns, select level
            if isinstance(data.columns, pd.MultiIndex):
                adj = data['Adj Close']
            else:
                adj = data['Adj Close'] if 'Adj Close' in data else data
        elif 'Close' in data.columns:
            if isinstance(data.columns, pd.MultiIndex):
                adj = data['Close']
            else:
                adj = data['Close']
        else:
            # Single-column Series?
            adj = data
        # Make sure adj is a DataFrame with columns as tickers
        if isinstance(adj, pd.Series):
            adj = adj.to_frame(name=tickers[0])
        # Ensure columns are strings (tickers)
        adj.columns = [str(col) for col in adj.columns]
        return adj
    except Exception as e:
        st.error(f"An error occurred while fetching stock data: {e}")
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
            st.warning("No macroeconomic data found for that series.")
            return pd.DataFrame()
        observations = data['observations']
        rows = []
        for obs in observations:
            date = obs.get('date')
            val = obs.get('value', '')
            # Skip missing or '.' values
            try:
                valf = float(val)
            except:
                continue
            rows.append({'date': date, 'value': valf})
        if not rows:
            st.warning("FRED returned observations but none were numeric.")
            return pd.DataFrame()
        macro_data = pd.DataFrame(rows)
        macro_data['date'] = pd.to_datetime(macro_data['date'])
        macro_data.set_index('date', inplace=True)
        return macro_data
    except Exception as e:
        st.warning(f"An error occurred when fetching macro data from FRED: {e}")
        return pd.DataFrame()

# Function to adjust returns based on macro data (if provided)
def adjust_returns_based_on_macro(mean_log_returns, macro_data):
    if macro_data is None or macro_data.empty:
        return mean_log_returns
    normalized_macro = (macro_data - macro_data.mean()) / macro_data.std()
    try:
        latest_macro_value = normalized_macro.iloc[-1].values[0]
    except Exception:
        return mean_log_returns
    # Scale mean returns modestly by the latest macro normalized deviation
    # Clip extreme adjustments
    factor = np.clip(1 + latest_macro_value, 0.5, 1.5)
    return mean_log_returns * factor

# Function to simulate stock prices with multivariate shocks
def simulate_stock_prices(historical_prices, num_simulations, num_days, macro_data=None):
    if historical_prices is None or historical_prices.empty:
        st.error("Historical prices are empty. Cannot simulate.")
        return {}
    log_returns = calculate_log_returns(historical_prices)
    if log_returns.empty:
        st.error("Not enough price history to compute returns.")
        return {}
    mean_log_returns = log_returns.mean()
    cov_log_returns = calculate_covariance_matrix(log_returns)

    # Adjust mean by macro if provided
    mean_log_returns = adjust_returns_based_on_macro(mean_log_returns, macro_data)

    symbols = list(historical_prices.columns)
    n_stocks = len(symbols)
    # Pre-allocate simulated arrays: shape (num_days, num_simulations)
    simulated_prices = {stock: np.zeros((num_days, num_simulations)) for stock in symbols}

    # set t=0 to today's (last observed) price
    last_prices = historical_prices.iloc[-1]
    for stock in symbols:
        simulated_prices[stock][0, :] = last_prices[stock]

    # convert to numpy arrays for multivariate draws
    mean_vec = mean_log_returns.reindex(symbols).fillna(0).values
    cov_mat = cov_log_returns.reindex(index=symbols, columns=symbols).fillna(0).values

    # Add a tiny jitter to cov_mat if not positive semi-definite to avoid numeric issues
    try:
        # Simulate day-by-day
        for t in range(1, num_days):
            # draws shape (num_simulations, n_stocks)
            random_shocks = np.random.multivariate_normal(mean_vec, cov_mat + 1e-10 * np.eye(n_stocks), size=num_simulations)
            # For each stock place the next price
            for i, stock in enumerate(symbols):
                simulated_prices[stock][t, :] = simulated_prices[stock][t - 1, :] * np.exp(random_shocks[:, i])
    except Exception as e:
        st.error(f"Simulation error: {e}")
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
    st.markdown("""
    Quick notes:
    - This app uses Yahoo Finance (`yfinance`) for historical prices and FRED for one macro series (CPI).
    - If Wikipedia structure changes, tickers fetch may silently return fewer tickers; you can manually input tickers in the sidebar.
    """)

    # Try to fetch sp500 tickers (non-blocking)
    sp500_tickers = get_sp500_tickers()
    st.sidebar.header('Simulation Parameters')

    default_tickers = ['AAPL', 'NVDA']
    # Allow manual input or multi-select from SP500
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
        st.warning("Please select at least one ticker or enter tickers manually.")
        st.stop()

    num_simulations = st.sidebar.slider('Number of Simulations', 100, 20000, 2000, step=100)
    num_days = st.sidebar.slider('Number of Days', 5, 252, 30, step=1)
    increase_percentage = st.sidebar.slider('Boom Return Threshold (fraction)', 0.0, 2.0, 0.2, step=0.01)
    decrease_percentage = st.sidebar.slider('Bust Return Threshold (fraction)', 0.0, 2.0, 0.2, step=0.01)

    period = st.sidebar.selectbox("Historical period", ["1y", "2y", "5y", "10y"], index=3)

    # Fetch historical prices
    historical_prices = fetch_stock_data(selected_tickers, period=period)
    if historical_prices is None or historical_prices.empty:
        st.error("No historical prices available. Adjust tickers or period.")
        st.stop()

    st.subheader('Historical Stock Prices (last 5 rows)')
    st.dataframe(historical_prices.tail())

    # Macroeconomic data fetch
    st.subheader('Macroeconomic Data (optional)')
    api_key = st.sidebar.text_input("FRED API Key (optional)", value="08828fc4fc9dbcfbea6f77718987ade3")
    series_id = st.sidebar.text_input("FRED Series ID", value="CPIAUCSL")
    macro_data = pd.DataFrame()
    if api_key and series_id:
        macro_data = fetch_macro_data_from_fred(api_key, series_id)
        if not macro_data.empty:
            st.write(macro_data.tail())
        else:
            st.info("No macro data available or parsing failed.")

    # Run simulation
    simulated_prices = simulate_stock_prices(historical_prices[selected_tickers], num_simulations, num_days, macro_data)
    if not simulated_prices:
        st.error("Simulation failed.")
        st.stop()

    st.subheader('Simulated Price Paths (sample of simulations)')
    for stock in selected_tickers:
        plt.figure(figsize=(10, 4))
        # Plot a sample of simulation paths (limit visual clutter)
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
