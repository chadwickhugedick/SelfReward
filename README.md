# Self-Rewarding Deep Reinforcement Learning (SRDRL) for Trading

This project implements the Self-Rewarding Double Deep Q-Network (SRDDQN) for financial trading strategies as described in the paper. The system combines reinforcement learning with a novel self-rewarding mechanism to optimize trading decisions.

## Project Structure

```
├── data/                  # Directory for storing financial datasets
├── models/                # Saved model checkpoints
├── src/                   # Source code
│   ├── data/              # Data processing modules
│   ├── environment/       # Trading environment simulation
│   ├── models/            # Neural network models
│   │   ├── feature_extraction/  # Time series feature extraction models
│   │   ├── reward_net/    # Self-rewarding network
│   │   └── agents/        # RL agents (DDQN, A2C, PPO)
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization tools
├── notebooks/             # Jupyter notebooks for experiments
├── results/               # Experimental results and figures
├── tests/                 # Unit tests
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd TraeML

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run training
python src/train.py --config configs/srddqn_config.yaml

# Run evaluation
python src/evaluate.py --model_path models/srddqn_model.pth --dataset data/test_data.csv
```

## Components

1. **Data Processing**: Handles financial time series data preprocessing and feature engineering.
2. **Trading Environment**: Simulates a trading environment with realistic constraints.
3. **Reward Network**: Implements the self-rewarding mechanism with time-series feature extraction models.
4. **Double DQN**: Implements the reinforcement learning agent with Q-Network and Target Network.
5. **Training Algorithm**: Implements the SRDDQN training algorithm as described in the paper.

## References

This implementation is based on the paper "Self-Rewarding Deep Reinforcement Learning for Financial Trading Strategies".# SelfReward

## Expert Selection

The configuration key `training.expert_selection` controls which precomputed expert reward the environment should prefer when available. Valid values and behavior:

- `null` / `None`: Default. Use the plain `expert_Min-Max` column if present (backwards compatible).
- `'best'`: Use the precomputed maximum across expert metrics for the configured lookahead/window (column `expert_best_{k}`). This is useful when you want the environment to adopt the strongest expert signal automatically.
- `'Min-Max'`, `'Return'`, or `'Sharpe'`: Force a specific expert metric to be used as the expert reward.

Examples:

- Prefer the strongest expert automatically:

```yaml
training:
	expert_selection: 'best'
```

- Force the environment to always use the Sharpe-like expert label:

```yaml
training:
	expert_selection: 'Sharpe'
```

If the requested expert column is not present in the processed dataset, the environment will fall back to available precomputed expert columns or to the online (runtime) reward computation.
