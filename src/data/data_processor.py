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
        Initialize the data processor with configuration
        """
        self.config = config
        self.scaler = MinMaxScaler()

    def load_parquet_data(self, file_path):
        """
        Load financial data from a parquet file and standardize format.
        
        Args:
            file_path (str): Path to the parquet file
            
        Returns:
            pd.DataFrame: DataFrame with standardized OHLCV data
        """
        try:
            print(f"Loading parquet data from {file_path}")
            data = pd.read_parquet(file_path)
            
            print(f"Loaded {len(data)} rows of data")
            print("Original columns:", data.columns.tolist())
            
            # Standardize column names
            data = self._standardize_columns(data)
            
            # Ensure datetime index
            data = self._ensure_datetime_index(data)
            
            # Validate and clean data
            data = self._validate_ohlcv_data(data)
            
            print(f"Final standardized data: {len(data)} rows")
            print("Standardized columns:", data.columns.tolist())
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading parquet data: {e}")
            return None

    def _standardize_columns(self, data):
        """
        Standardize column names to OHLCV format.
        
        Args:
            data (pd.DataFrame): Raw data with various column names
            
        Returns:
            pd.DataFrame: Data with standardized column names
        """
        # Expected columns for SRDDQN
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Preserve datetime columns for later processing
        datetime_columns = ['timestamp', 'date', 'datetime', 'time']
        preserved_datetime_cols = [col for col in datetime_columns if col in data.columns]
        
        # Auto-detect column names (handle different naming conventions)
        column_mapping = {}
        for col in data.columns:
            # Skip datetime columns during OHLCV mapping
            if col in datetime_columns:
                continue
                
            col_lower = col.lower().strip()
            if col_lower in ['open', 'o'] and 'Open' not in column_mapping.values():
                column_mapping[col] = 'Open'
            elif col_lower in ['high', 'h'] and 'High' not in column_mapping.values():
                column_mapping[col] = 'High'
            elif col_lower in ['low', 'l'] and 'Low' not in column_mapping.values():
                column_mapping[col] = 'Low'
            elif col_lower in ['close', 'c'] and 'Close' not in column_mapping.values():
                column_mapping[col] = 'Close'
            elif col_lower in ['volume', 'vol', 'v'] and 'Volume' not in column_mapping.values():
                column_mapping[col] = 'Volume'
        
        print("Detected column mapping:", column_mapping)
        
        # Rename columns to standard format
        if column_mapping:
            data = data.rename(columns=column_mapping)
        
        # Ensure we have required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns}")
            # Add Volume column if missing (crypto data sometimes lacks volume)
            if 'Volume' in missing_columns:
                data['Volume'] = 1.0
                print("Added default Volume column")
        
        # Keep required columns AND preserved datetime columns
        columns_to_keep = [col for col in required_columns if col in data.columns]
        columns_to_keep.extend(preserved_datetime_cols)
        data = data[columns_to_keep]
        
        if preserved_datetime_cols:
            print(f"Preserved datetime columns: {preserved_datetime_cols}")
        
        return data

    def _ensure_datetime_index(self, data):
        """
        Ensure the dataframe has a proper datetime index.
        
        Args:
            data (pd.DataFrame): Data possibly without datetime index
            
        Returns:
            pd.DataFrame: Data with datetime index
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            # Try common datetime column names
            datetime_columns = ['timestamp', 'date', 'datetime', 'time']
            datetime_col = None
            
            # Check both in columns and in index
            for col in datetime_columns:
                if col in data.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                print(f"Using '{datetime_col}' column as datetime index")
                try:
                    # Check sample of timestamp data
                    sample_value = data[datetime_col].iloc[0]
                    print(f"Sample timestamp value: {sample_value} (type: {type(sample_value)})")
                    
                    # Convert to datetime
                    datetime_series = pd.to_datetime(data[datetime_col])
                    data.index = datetime_series
                    data = data.drop(datetime_col, axis=1)
                    
                    print(f"Successfully converted to datetime index: {data.index.min()} to {data.index.max()}")
                    print(f"Index type: {type(data.index)}")
                    
                except Exception as e:
                    print(f"Failed to convert {datetime_col} to datetime: {e}")
                    print(f"Sample values: {data[datetime_col].head()}")
                    print("Keeping original index")
                    return data
            else:
                print("No datetime column found, using range index")
        else:
            print("Data already has datetime index")
        
        return data

    def resample_data(self, data, timeframe='5T'):
        """
        Resample high-frequency data to lower frequency.
        
        Args:
            data (pd.DataFrame): OHLCV data with datetime index
            timeframe (str): Pandas frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour)
            
        Returns:
            pd.DataFrame: Resampled OHLCV data
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            print("Warning: Data must have datetime index for resampling")
            return data
        
        print(f"Resampling from {len(data)} rows to {timeframe} bars...")
        
        # Define resampling rules for OHLCV data
        agg_dict = {
            'Open': 'first',    # First value in period
            'High': 'max',      # Maximum value in period
            'Low': 'min',       # Minimum value in period
            'Close': 'last',    # Last value in period
            'Volume': 'sum'     # Sum of volume
        }
        
        # Only aggregate columns that exist
        available_agg = {k: v for k, v in agg_dict.items() if k in data.columns}
        
        # Resample the data
        resampled_data = data.resample(timeframe).agg(available_agg).dropna()
        
        print(f"Resampled to {len(resampled_data)} {timeframe} bars")
        print(f"Data reduction: {len(data)} → {len(resampled_data)} ({len(resampled_data)/len(data)*100:.1f}%)")
        
        return resampled_data

    def _validate_ohlcv_data(self, data):
        """
        Validate and clean OHLCV data.
        
        Args:
            data (pd.DataFrame): OHLCV data to validate
            
        Returns:
            pd.DataFrame: Cleaned and validated data
        """
        original_length = len(data)
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Ensure positive prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Ensure High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        if all(col in data.columns for col in price_columns):
            valid_rows = (
                (data['High'] >= data['Low']) &
                (data['High'] >= data['Open']) & 
                (data['High'] >= data['Close']) &
                (data['Low'] <= data['Open']) &
                (data['Low'] <= data['Close'])
            )
            data = data[valid_rows]
        
        # Ensure Volume is non-negative
        if 'Volume' in data.columns:
            data = data[data['Volume'] >= 0]
        
        cleaned_length = len(data)
        if cleaned_length < original_length:
            print(f"Cleaned data: removed {original_length - cleaned_length} invalid rows")
        
        return data
    
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

    def compute_multi_horizon(self, data, horizons=None):
        """
        Compute multi-horizon aggregated features (e.g., daily, weekly) and return
        a DataFrame with prefixed column names aligned to the original index.

        Args:
            data (pd.DataFrame): Original OHLCV + indicator dataframe with DatetimeIndex
            horizons (dict): Mapping like {'daily':'1D','weekly':'1W'}

        Returns:
            pd.DataFrame: DataFrame with columns prefixed by horizon keys, reindexed to original index
        """
        if horizons is None:
            horizons = {'dh': '1D', 'wh': '1W'}

        if not isinstance(data.index, pd.DatetimeIndex):
            return pd.DataFrame(index=data.index)

        agg_results = []
        for prefix, freq in horizons.items():
            try:
                res = data.resample(freq).agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
            except Exception:
                # If core OHLCV columns missing, try identity resample
                res = data.resample(freq).last().dropna()

            # Add simple indicators on resampled data
            res[f'{prefix}_SMA_10'] = res['Close'].rolling(window=10, min_periods=1).mean()
            res[f'{prefix}_SMA_20'] = res['Close'].rolling(window=20, min_periods=1).mean()
            res[f'{prefix}_Returns'] = res['Close'].pct_change().fillna(0.0)
            # Keep only the new prefixed columns
            cols_to_keep = [c for c in res.columns if c.startswith(prefix)]
            # If res contains non-prefixed OHLCV, prefix them for merging
            for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if base in res.columns:
                    res[f'{prefix}_{base}'] = res[base]
                    cols_to_keep.append(f'{prefix}_{base}')

            out = res[cols_to_keep]

            # Reindex to original data index using forward-fill so each original timestamp gets latest horizon value
            out_reindexed = out.reindex(data.index, method='ffill')
            # Rename columns to ensure uniqueness (already prefixed)
            agg_results.append(out_reindexed)

        if agg_results:
            combined = pd.concat(agg_results, axis=1)
            # Fill any remaining NaNs with 0
            combined = combined.fillna(0.0)
            return combined

        return pd.DataFrame(index=data.index)
    
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
        Create sequences of data for time series prediction using vectorized operations.

        Args:
            data (pd.DataFrame): DataFrame with features
            window_size (int): Size of the window for sequence creation

        Returns:
            tuple: (X, y) where X is the sequence data and y is the target
        """
        # Convert to numpy array for efficient processing
        data_array = data.values
        num_samples = len(data_array) - window_size

        if num_samples <= 0:
            return np.array([]), np.array([])

        # Use stride tricks to create sliding windows efficiently
        # Create sliding windows along the time dimension
        X = np.lib.stride_tricks.sliding_window_view(data_array, window_size, axis=0)

        # Some numpy versions / memory layouts may produce windows with axes in
        # the order (samples, features, window) instead of (samples, window, features).
        # Normalize to (samples, window_size, n_features).
        if X.ndim == 3 and X.shape[1] == data_array.shape[1] and X.shape[2] == window_size:
            X = X.transpose(0, 2, 1)

        # Ensure we have exactly num_samples sequences (same as original loop)
        if X.shape[0] > num_samples:
            X = X[:num_samples]

        # Extract target values (next day's close price)
        y = data_array[window_size:, data.columns.get_loc('Close')]

        return X, y

    def create_sequences_chunked(self, data, window_size, chunk_size=10000):
        """
        Create sequences from large datasets using chunked processing to handle memory constraints.

        Args:
            data (pd.DataFrame): DataFrame with features
            window_size (int): Size of the window for sequence creation
            chunk_size (int): Size of chunks to process at once

        Returns:
            tuple: (X, y) where X is the sequence data and y is the target
        """
        data_array = data.values
        total_samples = len(data_array) - window_size

        if total_samples <= 0:
            return np.array([]), np.array([])

        # Calculate number of chunks needed
        num_chunks = int(np.ceil(total_samples / chunk_size))

        X_list = []
        y_list = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_samples)

            # Process chunk
            chunk_data = data_array[start_idx:end_idx + window_size]
            chunk_sequences = len(chunk_data) - window_size

            if chunk_sequences > 0:
                # Create sequences for this chunk
                X_chunk = np.lib.stride_tricks.sliding_window_view(chunk_data, window_size, axis=0)

                # Ensure correct format
                if X_chunk.ndim == 3:
                    # Normalize axis order if needed
                    if X_chunk.shape[1] == window_size:
                        pass  # Already in correct format
                    elif X_chunk.shape[1] == chunk_data.shape[1] and X_chunk.shape[2] == window_size:
                        X_chunk = X_chunk.transpose(0, 2, 1)
                
                # Ensure the number of windows equals chunk_sequences (sliding_window_view may return +1)
                if isinstance(X_chunk, np.ndarray) and X_chunk.ndim == 3:
                    if X_chunk.shape[0] > chunk_sequences:
                        X_chunk = X_chunk[:chunk_sequences]
                else:
                    # Fallback to manual creation
                    num_chunk_samples = len(chunk_data) - window_size
                    if num_chunk_samples > 0:
                        X_chunk = np.array([chunk_data[i:i+window_size] for i in range(num_chunk_samples)])
                    else:
                        X_chunk = np.array([])

                # Extract targets for this chunk
                y_chunk = chunk_data[window_size:, data.columns.get_loc('Close')]

                X_list.append(X_chunk)
                y_list.append(y_chunk)

        # Concatenate all chunks
        if X_list:
            X = np.concatenate(X_list, axis=0)
            y = np.concatenate(y_list, axis=0)
        else:
            X = np.array([])
            y = np.array([])

        # Trim to the exact expected number of samples to avoid overlaps at chunk boundaries
        if isinstance(X, np.ndarray) and X.size and total_samples is not None:
            if X.shape[0] > total_samples:
                X = X[:total_samples]
        if isinstance(y, np.ndarray) and y.size and total_samples is not None:
            if y.shape[0] > total_samples:
                y = y[:total_samples]

        return X, y

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
    
    def prepare_data_from_parquet(self, file_path, resample_timeframe=None):
        """
        Prepare data for the SRDDQN model from a parquet file.
        
        Args:
            file_path (str): Path to the parquet file
            resample_timeframe (str): Optional resampling timeframe (e.g., '5T', '15T', '1H')
            
        Returns:
            tuple: (train_data, val_data, test_data) - processed and ready for training
        """
        # Load parquet data
        data = self.load_parquet_data(file_path)
        
        if data is None:
            print("Failed to load parquet data")
            return None, None, None
        
        print(f"Loaded {len(data)} rows from parquet file")
        
        # Resample FIRST if requested (before any other processing)
        if resample_timeframe:
            # Make sure we have datetime index for resampling
            if not isinstance(data.index, pd.DatetimeIndex):
                print("Error: Cannot resample without datetime index")
                return None, None, None
            
            data = self.resample_data(data, resample_timeframe)
            if len(data) == 0:
                print("No data remaining after resampling")
                return None, None, None
        
        # Add technical indicators
        data_with_indicators = self.add_technical_indicators(data)

        # Compute multi-horizon aggregated features if enabled in config
        mh_config = self.config.get('data', {}).get('multi_horizon', {})
        mh_enabled = mh_config.get('enabled', True) if isinstance(mh_config, dict) else bool(mh_config)
        mh_horizons = mh_config.get('horizons') if isinstance(mh_config, dict) else None
        if mh_enabled:
            try:
                horizon_feats = self.compute_multi_horizon(data_with_indicators, horizons=mh_horizons)
                if not horizon_feats.empty:
                    # Work on a copy to avoid SettingWithCopyWarning when assigning new columns
                    data_with_indicators = data_with_indicators.copy()
                    # Merge horizon features into main dataframe (align by index)
                    # Avoid column collisions by only adding columns that do not exist
                    for col in horizon_feats.columns:
                        if col not in data_with_indicators.columns:
                            # Use .loc to ensure assignment happens on the DataFrame and not a view
                            data_with_indicators.loc[:, col] = horizon_feats[col]
            except Exception as e:
                print(f"Warning: failed to compute multi-horizon features: {e}")
        
        # Split data into train/val/test
        train_ratio = self.config.get('data', {}).get('train_ratio', 0.7)
        val_ratio = self.config.get('data', {}).get('val_ratio', 0.15)
        
        train_data, val_data, test_data = self.split_data(
            data_with_indicators, train_ratio, val_ratio
        )
        
        print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Precompute expert labels (vectorized) using lookahead equal to window_size if available
        # Determine lookaheads to precompute (configurable)
        window_size = self.config.get('environment', {}).get('window_size', 20)
        # Training may specify multiple lookaheads under training.lookaheads
        lookaheads = None
        try:
            lookaheads = self.config.get('training', {}).get('lookaheads', None)
        except Exception:
            lookaheads = None

        # Default to single lookahead = window_size if not provided
        if not lookaheads:
            lookaheads = [window_size]

        try:
            train_labels = self.compute_expert_labels_multi(train_data, lookaheads)
            val_labels = self.compute_expert_labels_multi(val_data, lookaheads)
            test_labels = self.compute_expert_labels_multi(test_data, lookaheads)
        except Exception:
            train_labels = None
            val_labels = None
            test_labels = None

        # Post-process label DataFrames: add best_{k} = max across expert metrics for each lookahead
        # and add plain Min-Max/Return/Sharpe columns corresponding to configured window_size for backwards compatibility
        def postprocess_labels(labels_df, lookaheads_list, window_k):
            if labels_df is None:
                return None

            # Ensure lookaheads_list is iterable
            if isinstance(lookaheads_list, int):
                lookaheads_list = [lookaheads_list]

            for k in lookaheads_list:
                col_min = f'Min-Max_{k}'
                col_ret = f'Return_{k}'
                col_shp = f'Sharpe_{k}'
                # If the three metric columns exist, compute best_{k}
                present = [c for c in (col_min, col_ret, col_shp) if c in labels_df.columns]
                if present:
                    # Use numpy maximum reduce, align missing columns as zeros
                    arrs = []
                    for c in (col_min, col_ret, col_shp):
                        if c in labels_df.columns:
                            arrs.append(labels_df[c].values)
                        else:
                            arrs.append(np.zeros(len(labels_df)))
                    best_vals = np.maximum.reduce(arrs)
                    labels_df[f'best_{k}'] = best_vals

            # Create plain expert columns for the chosen window_k: 'Min-Max', 'Return', 'Sharpe'
            chosen = None
            if window_k in lookaheads_list:
                chosen = window_k
            elif len(lookaheads_list) > 0:
                chosen = lookaheads_list[0]

            if chosen is not None:
                for base in ['Min-Max', 'Return', 'Sharpe']:
                    suff = f"{base}_{chosen}"
                    if suff in labels_df.columns:
                        labels_df[base] = labels_df[suff].values
                    else:
                        labels_df[base] = np.zeros(len(labels_df))

            # Fill NaNs
            labels_df = labels_df.fillna(0.0)
            return labels_df

        train_labels = postprocess_labels(train_labels, lookaheads, window_size)
        val_labels = postprocess_labels(val_labels, lookaheads, window_size)
        test_labels = postprocess_labels(test_labels, lookaheads, window_size)

        # Normalize the data (fit scaler on training data)
        train_data_normalized = self.normalize_data(train_data)

        # Apply same normalization to val and test data
        val_data_normalized = pd.DataFrame(
            self.scaler.transform(val_data),
            columns=val_data.columns,
            index=val_data.index
        )

        test_data_normalized = pd.DataFrame(
            self.scaler.transform(test_data),
            columns=test_data.columns,
            index=test_data.index
        )

        # Attach expert labels as additional columns (if available)
        def attach_labels(df_norm, labels_df):
            if labels_df is None:
                return df_norm
            # Align by index and add prefixed columns
            for col in labels_df.columns:
                df_norm[f"expert_{col}"] = labels_df[col].reindex(df_norm.index).fillna(0.0)
            return df_norm

        train_data_normalized = attach_labels(train_data_normalized, train_labels)
        val_data_normalized = attach_labels(val_data_normalized, val_labels)
        test_data_normalized = attach_labels(test_data_normalized, test_labels)

        return train_data_normalized, val_data_normalized, test_data_normalized

    def compute_expert_labels(self, df, lookahead=20):
        """
        Compute expert labels (Min-Max, forward Return, Sharpe-like) using vectorized operations.

        Args:
            df (pd.DataFrame): DataFrame with a 'Close' column and datetime index
            lookahead (int): Number of future steps to look ahead

        Returns:
            pd.DataFrame: DataFrame with columns ['Min-Max', 'Return', 'Sharpe'] aligned to df.index
        """
        if 'Close' not in df.columns or len(df) == 0:
            return None

        close = df['Close'].values
        n = len(close)
        if lookahead <= 0:
            lookahead = 1

        # If not enough points, return zeros
        if n <= 1:
            labels = pd.DataFrame({
                'Min-Max': np.zeros(n),
                'Return': np.zeros(n),
                'Sharpe': np.zeros(n)
            }, index=df.index)
            return labels

        # Use sliding windows to compute future windows
        # windows shape will be (n - lookahead, lookahead + 1) when including current
        if n > lookahead:
            try:
                windows = np.lib.stride_tricks.sliding_window_view(close, lookahead + 1)
            except Exception:
                # Fallback: construct with loop if stride trick unavailable
                windows = np.array([close[i:i + lookahead + 1] for i in range(n - lookahead)])

            # For each window, current price is window[0], future prices window[1:]
            current = windows[:, 0]
            future = windows[:, 1:]

            # Forward return: price at t+lookahead vs current
            forward = windows[:, -1]
            forward_return = (forward - current) / (current + 1e-12)

            # Min-Max: range in future window normalized by current price
            max_future = np.max(future, axis=1)
            min_future = np.min(future, axis=1)
            min_max = (max_future - min_future) / (current + 1e-12)
            # Scale with tanh for boundedness
            min_max_scaled = np.tanh(min_max * 2.0)

            # Sharpe-like: mean/std of future percentage changes
            # Compute returns between consecutive future points
            future_returns = (future[:, 1:] - future[:, :-1]) / (future[:, :-1] + 1e-12)
            mean_fr = np.mean(future_returns, axis=1)
            std_fr = np.std(future_returns, axis=1)
            sharpe_like = np.where(std_fr > 0, mean_fr / std_fr, 0.0)

            # Build result arrays aligned to original index
            min_max_full = np.zeros(n)
            return_full = np.zeros(n)
            sharpe_full = np.zeros(n)

            min_max_full[: n - lookahead] = min_max_scaled
            return_full[: n - lookahead] = forward_return
            sharpe_full[: n - lookahead] = sharpe_like
        else:
            # Not enough future points; return zeros
            min_max_full = np.zeros(n)
            return_full = np.zeros(n)
            sharpe_full = np.zeros(n)

        labels = pd.DataFrame({
            'Min-Max': min_max_full,
            'Return': return_full,
            'Sharpe': sharpe_full
        }, index=df.index)

        return labels

    def compute_expert_labels_multi(self, df, lookaheads=None):
        """
        Compute expert labels for multiple lookahead horizons and return a combined DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with 'Close' and datetime index
            lookaheads (list[int] or int): list of lookahead integers or single int

        Returns:
            pd.DataFrame: DataFrame with columns like 'Min-Max_{k}', 'Return_{k}', 'Sharpe_{k}'
        """
        if lookaheads is None:
            return self.compute_expert_labels(df)

        if isinstance(lookaheads, int):
            lookaheads = [lookaheads]

        parts = []
        for k in lookaheads:
            lbl = self.compute_expert_labels(df, lookahead=k)
            # rename columns to include suffix
            renamed = lbl.rename(columns={
                'Min-Max': f'Min-Max_{k}',
                'Return': f'Return_{k}',
                'Sharpe': f'Sharpe_{k}'
            })
            parts.append(renamed)

        if parts:
            combined = pd.concat(parts, axis=1)
            # fill NaNs with zeros
            combined = combined.fillna(0.0)
            return combined

        return None
    
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