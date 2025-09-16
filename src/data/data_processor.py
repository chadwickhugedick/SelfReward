import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import ta

class DataProcessor:
    """
    Class for processing financial time series data for the SRDDQN model.
    Handles data downloading, preprocessing, feature engineering, and splitting.
    """
    
    def __init__(self, config):
        """
        Initialize the data processor with configuration.
        
        Args:
            config (dict): Configuration dictionary with data settings
        """
        self.config = config
        self.scaler = MinMaxScaler()
        
    def download_data(self, ticker, start_date, end_date):
        """
        Download financial data for a given ticker and date range.

        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format

        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            print(f"Downloaded {len(data)} rows of data for {ticker}")

            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)  # Drop the ticker level

            return data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def download_multi_ticker_data(self, tickers, start_date, end_date):
        """
        Download financial data for multiple tickers and combine them.

        Args:
            tickers (list): List of stock ticker symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format

        Returns:
            dict: Dictionary with ticker as key and DataFrame as value
        """
        data_dict = {}
        for ticker in tickers:
            data = self.download_data(ticker, start_date, end_date)
            if data is not None:
                data_dict[ticker] = data
        return data_dict
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to the dataset.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        # Make sure data is a DataFrame with proper column names
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            # Add moving averages
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
            
            # Add momentum indicators
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            data['MACD'] = ta.trend.macd(data['Close'])
            data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
            
            # Add volatility indicators
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
            data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
            
            # Add volume indicators
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
        else:
            print("Error: Input must be a DataFrame with 'Close' column")
            return data
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    
    def normalize_data(self, data):
        """
        Normalize the data using Min-Max scaling.
        
        Args:
            data (pd.DataFrame): DataFrame with features
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        # Store the dates and actual close prices before normalization
        dates = data.index
        close_prices = data['Close'].values
        
        # Get feature names
        feature_names = data.columns
        
        # Normalize the data
        normalized_data = self.scaler.fit_transform(data)
        
        # Convert back to DataFrame
        normalized_df = pd.DataFrame(normalized_data, columns=feature_names, index=dates)
        
        # Store the original close prices for later use
        self.original_close_prices = close_prices
        
        return normalized_df
    
    def create_sequences(self, data, window_size):
        """
        Create sequences of data for time series prediction.
        
        Args:
            data (pd.DataFrame): DataFrame with features
            window_size (int): Size of the window for sequence creation
            
        Returns:
            tuple: (X, y) where X is the sequence data and y is the target
        """
        X, y = [], []
        
        for i in range(len(data) - window_size):
            X.append(data.iloc[i:i+window_size].values)
            # Target is the next day's close price
            y.append(data.iloc[i+window_size]['Close'])
        
        return np.array(X), np.array(y)
    
    def split_data(self, data, train_ratio=0.7, val_ratio=0.15):
        """
        Split the data into training, validation, and testing sets.
        
        Args:
            data (pd.DataFrame): DataFrame with features
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size+val_size]
        test_data = data.iloc[train_size+val_size:]
        
        return train_data, val_data, test_data
    
    def prepare_data(self, ticker):
        """
        Prepare data for the SRDDQN model.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            tuple: (train_X, train_y, test_X, test_y)
        """
        # Download training data
        train_data = self.download_data(
            ticker, 
            self.config['data']['train_start_date'], 
            self.config['data']['train_end_date']
        )
        
        # Download testing data
        test_data = self.download_data(
            ticker, 
            self.config['data']['test_start_date'], 
            self.config['data']['test_end_date']
        )
        
        if train_data is None or test_data is None:
            return None
        
        # Add technical indicators
        train_data = self.add_technical_indicators(train_data)
        test_data = self.add_technical_indicators(test_data)
        
        # Normalize the data
        train_data_normalized = self.normalize_data(train_data)
        
        # Use the same scaler for test data to ensure consistency
        test_data_normalized = pd.DataFrame(
            self.scaler.transform(test_data),
            columns=test_data.columns,
            index=test_data.index
        )
        
        # Create sequences
        window_size = self.config['environment']['window_size']
        train_X, train_y = self.create_sequences(train_data_normalized, window_size)
        test_X, test_y = self.create_sequences(test_data_normalized, window_size)
        
        return train_X, train_y, test_X, test_y, train_data, test_data

    def prepare_multi_ticker_data(self, tickers=None):
        """
        Prepare data for multiple tickers for the SRDDQN model.

        Args:
            tickers (list): List of stock ticker symbols. If None, uses config tickers.

        Returns:
            dict: Dictionary with ticker as key and processed data as value
        """
        if tickers is None:
            tickers = self.config['data'].get('tickers', [self.config['data']['ticker']])

        # Download data for all tickers
        data_dict = self.download_multi_ticker_data(
            tickers,
            self.config['data']['train_start_date'],
            self.config['data']['test_end_date']
        )

        processed_data = {}
        for ticker, data in data_dict.items():
            if data is not None:
                # Add technical indicators
                data_with_indicators = self.add_technical_indicators(data)

                # Split data
                train_data, val_data, test_data = self.split_data(data_with_indicators)

                # Normalize training data
                train_normalized = self.normalize_data(train_data)

                # Use same scaler for validation and test data
                val_normalized = pd.DataFrame(
                    self.scaler.transform(val_data),
                    columns=val_data.columns,
                    index=val_data.index
                )
                test_normalized = pd.DataFrame(
                    self.scaler.transform(test_data),
                    columns=test_data.columns,
                    index=test_data.index
                )

                # Create sequences
                window_size = self.config['environment']['window_size']
                train_X, train_y = self.create_sequences(train_normalized, window_size)
                val_X, val_y = self.create_sequences(val_normalized, window_size)
                test_X, test_y = self.create_sequences(test_normalized, window_size)

                processed_data[ticker] = {
                    'train_X': train_X, 'train_y': train_y,
                    'val_X': val_X, 'val_y': val_y,
                    'test_X': test_X, 'test_y': test_y,
                    'train_data': train_data, 'val_data': val_data, 'test_data': test_data
                }

        return processed_data