# BTC Trading with SRDDQN - Quick Start Guide

## 🚀 Your system is now ready for BTC 1-minute data!

### Step 1: Place your BTC parquet file
- Copy your BTC 1-minute parquet file to the TraeML directory
- Example: `btc_1min_data.parquet`

### Step 2: Pre-train the reward network
```bash
python main.py --parquet_file btc_1min_data.parquet --pretrain
```

### Step 3: Train the SRDDQN agent
```bash
python main.py --parquet_file btc_1min_data.parquet --train
```

### Step 4: Evaluate performance
```bash
python main.py --parquet_file btc_1min_data.parquet --evaluate
```

## 📊 Expected Parquet File Format

Your parquet file should have columns like:
- `open`, `high`, `low`, `close`, `volume` (any case)
- OR `Open`, `High`, `Low`, `Close`, `Volume`
- Optional: `timestamp`, `date`, `datetime` for time index

The system automatically detects and standardizes column names!

## ⚡ Performance Optimizations for Crypto

For faster training on 1-minute crypto data:
1. **Reduce window size** in config: `window_size: 10` (instead of 20)
2. **Smaller episodes**: `num_episodes: 10` for quick tests
3. **Keep NLinear model**: Already optimized for speed

## 🎯 Expected Results

With 1-minute BTC data, you might see:
- **Faster episode completion**: ~10-15 seconds per episode
- **High volatility learning**: Agent learns from BTC price swings
- **Different patterns**: Crypto behaves differently than stocks

## 🔧 Configuration Adjustments for Crypto

Edit `configs/srddqn_config.yaml`:

```yaml
environment:
  initial_capital: 10000    # Smaller starting capital for crypto
  transaction_cost: 0.001   # Lower fees for crypto exchanges

training:
  num_episodes: 15          # Adjust based on data size
  
data:
  train_ratio: 0.8         # Use more data for training
  val_ratio: 0.1           # Less for validation
```

## 📈 What the System Does Automatically

1. **Column Detection**: Finds OHLCV columns regardless of naming
2. **Data Validation**: Removes invalid price data
3. **Technical Indicators**: Adds RSI, MACD, Bollinger Bands, etc.
4. **Normalization**: Scales all features for neural network training
5. **Time-aware Splitting**: Maintains chronological order

## 🚨 Troubleshooting

### "No datetime index found"
- Add a timestamp column to your parquet file
- Or the system will use row numbers (still works!)

### "Missing columns"
- Ensure your parquet has OHLC columns
- Volume is optional (system adds default if missing)

### Training too slow
- Reduce `num_episodes` to 5-10 for testing
- Use smaller `window_size` (10-15)
- Reduce `batch_size` to 8-16

### Memory issues
- Reduce `replay_buffer_size` to 1000
- Use smaller `hidden_size` (16-32)

## 💡 Pro Tips

1. **Test first**: Start with 5 episodes to verify everything works
2. **Monitor performance**: Check the results/ directory for charts
3. **Experiment**: Try different reward functions in config
4. **Compare timeframes**: Test 5-minute or hourly data for different patterns

Ready to trade Bitcoin with AI! 🚀