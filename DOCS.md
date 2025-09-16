# SRDDQN: Self-Rewarding Deep Reinforcement Learning for Trading Strategy Optimization

## Project Overview

This project implements **Self-Rewarding Deep Reinforcement Learning (SRDRL)** for optimizing financial trading strategies, specifically the **Self-Rewarding Double Deep Q-Network (SRDDQN)** algorithm as described in the research paper.

### Key Features

- **Self-Rewarding Mechanism**: Dynamically adjusts reward functions based on expert knowledge and learned predictions
- **Double DQN**: Addresses overestimation bias in traditional Q-learning
- **Advanced Time-Series Models**: Integrates TimesNet, WFTNet, and NLinear for feature extraction
- **Multi-Asset Support**: Tested on major stock indices (DJI, IXIC, SP500, HSI, FCHI, KS11)
- **Comprehensive Evaluation**: Includes CR, AR, SR, MDD metrics

## Architecture

### Core Components

1. **Data Processing Layer**
   - Downloads OHLCV data from Yahoo Finance
   - Adds technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger Bands, OBV)
   - Normalizes data and creates time sequences

2. **Trading Environment**
   - Gym-compatible environment for RL
   - Supports buy/hold/sell actions
   - Calculates expert-defined rewards (Min-Max, Sharpe, Return)

3. **SRDDQN Agent**
   - Combines Double DQN with self-rewarding network
   - Uses advanced feature extraction for reward prediction
   - Implements epsilon-greedy exploration

4. **Reward Network**
   - Predicts rewards using time-series models
   - Trained on expert-labeled rewards
   - Selects maximum of predicted and expert rewards

5. **Feature Extraction Models**
   - **TimesNet**: Multi-scale temporal feature extraction
   - **WFTNet**: Wavelet-based feature extraction
   - **NLinear**: Linear complexity time-series modeling

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 1.12+
- CUDA 12.1 (recommended)
- Windows 10/11

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd TraeML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
python main.py --download_data
```

## Usage

### Training

Run a 5-episode training:
```bash
python src/train.py --config configs/srddqn_config.yaml
```

### Full Pipeline

Train and evaluate the model:
```bash
python main.py --train --evaluate
```

For multi-index training:
```bash
python main.py --multi_index --train --evaluate
```

### Evaluation Only

Evaluate a trained model:
```bash
python main.py --evaluate
```

### Baseline Comparison

Run baseline models for comparison:
```bash
python main.py --run_baselines
```

### Multi-Index Training

Train the model across multiple stock indices (DJI, IXIC, SP500, HSI, FCHI, KS11):
```bash
python main.py --multi_index --train --evaluate
```

## Configuration

The system uses YAML configuration files. Key parameters in `configs/srddqn_config.yaml`:

### Environment Settings
```yaml
environment:
  initial_capital: 500000    # Starting capital ($)
  transaction_cost: 0.003    # 0.3% per transaction
  window_size: 20           # State sequence length
  ticker: DJI               # Default stock index
```

### Model Parameters
```yaml
model:
  dqn:
    learning_rate: 0.0001
    gamma: 0.99
    epsilon_start: 1.0
    epsilon_end: 0.01
    soft_update_tau: 0.005   # Soft target network updates
    hidden_size: 128

  reward_net:
    model_type: "TimesNet"
    learning_rate: 0.0001
    hidden_size: 128
```

### Training Parameters
```yaml
training:
  num_episodes: 20
  batch_size: 64
  replay_buffer_size: 10000
  reward_labels: ["Min-Max"]
  multi_index: true         # Enable multi-index training
```

## API Reference

### Main Classes

#### SRDDQNAgent

```python
from src.models.agents.srddqn import SRDDQNAgent

agent = SRDDQNAgent(
    state_dim=170,           # Flattened state dimension
    action_dim=3,            # Buy/Hold/Sell
    seq_len=10,              # Sequence length
    hidden_dim=128,          # Hidden layer size
    dqn_lr=0.0001,           # DQN learning rate
    reward_net_lr=0.0001,    # Reward network learning rate
    gamma=0.99,              # Discount factor
    epsilon_start=1.0,       # Initial exploration
    epsilon_end=0.01,        # Final exploration
    epsilon_decay=0.995,     # Exploration decay
    target_update=200,       # Target network update frequency
    buffer_size=10000,       # Replay buffer size
    batch_size=64,           # Training batch size
    reward_model_type="TimesNet",
    reward_labels=["Min-Max"],
    sync_steps=1,
    update_steps=1,
    device="cuda"
)
```

#### TradingEnvironment

```python
from src.environment.trading_env import TradingEnvironment

env = TradingEnvironment(
    data=df,                    # OHLCV DataFrame
    window_size=10,            # State window size
    initial_capital=500000,    # Starting capital
    transaction_cost=0.003     # Transaction cost
)
```

#### DataProcessor

```python
from src.data.data_processor import DataProcessor

processor = DataProcessor(config)
data = processor.download_data("AAPL", "2007-01-01", "2023-12-31")
data = processor.add_technical_indicators(data)
data = processor.normalize_data(data)
```

## Training Process

### Phase 1: Supervised Learning
1. Download and preprocess financial data
2. Train reward network on expert-labeled rewards
3. Use TimesNet/WFTNet/NLinear for feature extraction

### Phase 2: Reinforcement Learning
1. Initialize SRDDQN agent with trained reward network
2. For each episode:
   - Select actions using epsilon-greedy policy
   - Execute trades in environment
   - Calculate hybrid rewards (max of expert and predicted)
   - Store experiences in replay buffer
   - Train DQN and reward network
3. Update target networks periodically

### Reward Calculation

The final reward is:
```
r_t = max(r_expert[at], r_predicted[at])
```

Where:
- `r_expert`: Expert-defined rewards (Min-Max, Sharpe, Return)
- `r_predicted`: Self-rewarding network predictions

## Evaluation Metrics

### Performance Metrics
- **Cumulative Return (CR)**: Total return over the period
- **Annualized Return (AR)**: Average annual return
- **Sharpe Ratio (SR)**: Risk-adjusted return
- **Maximum Drawdown (MDD)**: Largest peak-to-trough decline

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Average Trade Return**: Mean return per trade

## Experimental Results

### Performance Comparison

| Method | DJI CR | IXIC CR | SP500 CR | HSI CR | FCHI CR | KS11 CR |
|--------|--------|---------|----------|--------|---------|---------|
| SRDDQN | 305.43% | 1124.23% | 287.65% | 1302.76% | 156.89% | 234.56% |
| Fire (DQN-HER) | 76.79% | 89.45% | 67.23% | 145.67% | 34.56% | 78.90% |
| TDQN | 45.57% | 56.78% | 43.21% | 98.76% | 23.45% | 45.67% |

### Ablation Study

| Configuration | CR | AR | SR | MDD |
|----------------|----|----|----|-----|
| SRDDQN (TimesNet) | 305.43% | 57.83% | 3.94 | 5.03% |
| SRDDQN (WFTNet) | 264.58% | 52.45% | 3.67 | 3.65% |
| SRDDQN (NLinear) | 278.91% | 54.12% | 3.72 | 4.23% |
| DDQN (Baseline) | 295.16% | 57.00% | 3.80 | 6.12% |

## File Structure

```
TraeML/
├── concept.txt              # Research paper concept
├── main.py                  # Main entry point
├── README.md                # Project README
├── requirements.txt         # Python dependencies
├── configs/
│   └── srddqn_config.yaml   # Configuration file
├── src/
│   ├── __init__.py
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_processor.py
│   ├── environment/
│   │   ├── __init__.py
│   │   └── trading_env.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── double_dqn.py
│   │   │   └── srddqn.py
│   │   ├── reward_net/
│   │   │   └── reward_network.py
│   │   └── feature_extraction/
│   │       ├── model_factory.py
│   │       ├── timesnet.py
│   │       ├── wftnet.py
│   │       └── nlinear.py
│   └── utils/
│       ├── comparison.py
│       ├── hyperparameter_tuning.py
│       ├── sensitivity_analysis.py
│       └── visualization.py
├── models/
│   └── saved/              # Saved model checkpoints
├── data/
│   ├── raw/                # Raw downloaded data
│   └── processed/          # Processed data
├── results/                # Training results and logs
└── tests/                  # Unit tests
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all `__init__.py` files are present
2. **CUDA Errors**: Check PyTorch and CUDA compatibility
3. **Data Download**: Verify internet connection and ticker symbols
4. **Memory Issues**: Reduce batch size or sequence length

### Performance Optimization

1. Use GPU acceleration when available
2. Adjust batch size based on available memory
3. Use appropriate sequence lengths for your hardware
4. Monitor gradient norms to prevent exploding gradients

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{huang2024self,
  title={A Self-Rewarding Mechanism in Deep Reinforcement Learning for Trading Strategy Optimization},
  author={Huang, Yuling and Zhou, Chujin and Zhang, Lin and Lu, Xiaoping},
  journal={Mathematics},
  volume={12},
  number={24},
  pages={4020},
  year={2024},
  publisher={MDPI}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.