import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

from src.data.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment
from src.models.reward_net.reward_network import RewardNetwork
from src.utils.device_utils import get_device


class RewardPretrainingDataset(Dataset):
    """Dataset for pre-training the reward network with expert-labeled rewards."""
    
    def __init__(self, data, window_size, reward_labels=['Min-Max'], expert_selection=None):
        """
        Initialize the dataset.
        
        Args:
            data (pd.DataFrame): Market data
            window_size (int): Size of the observation window
            reward_labels (list): List of reward labels to use
        """
        self.data = data
        self.window_size = window_size
        self.reward_labels = reward_labels
        
        # Create a temporary environment to generate expert rewards
        self.env = TradingEnvironment(
            data=data,
            window_size=window_size,
            initial_capital=500000,
            transaction_cost=0.003,
            expert_selection=expert_selection
        )
        
        # Generate all possible state-action-reward tuples
        self.samples = self._generate_samples()
        
    def _generate_samples(self):
        """Generate state-action-reward samples for supervised learning."""
        samples = []
        
        print("Generating expert-labeled samples for pre-training...")

        # Reset environment (Gymnasium-style)
        state, info = self.env.reset()

        for step in tqdm(range(len(self.data) - self.window_size - 1)):
            if self.env.done:
                break
                
            # Get current state sequence (reshape for feature extraction)
            feature_dim = state.shape[1] if len(state.shape) > 1 else len(state) // self.window_size
            state_sequence = state.reshape(self.window_size, feature_dim)
            
            # Try all possible actions and collect expert rewards
            for action in range(3):  # 0: Hold, 1: Buy, 2: Sell
                # Save current environment state
                env_backup = {
                    'current_step': self.env.current_step,
                    'capital': self.env.capital,
                    'shares_held': self.env.shares_held,
                    'current_position': self.env.current_position,
                    'portfolio_values': self.env.portfolio_values.copy(),
                    'returns': self.env.returns.copy(),
                    'portfolio_peak': self.env.portfolio_peak,
                    'max_drawdown': self.env.max_drawdown,
                    'current_drawdown': self.env.current_drawdown
                }
                
                # Take action and get expert reward (Gymnasium-style)
                _, expert_reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)
                reward_dict = info.get('reward_dict', {})
                
                # Store sample for each reward label
                for reward_label in self.reward_labels:
                    if reward_label in reward_dict:
                        expert_reward_val = reward_dict[reward_label]
                        
                        # Skip samples with NaN or inf expert rewards
                        if np.isnan(expert_reward_val) or np.isinf(expert_reward_val):
                            continue
                            
                        samples.append({
                            'state_sequence': state_sequence.copy(),
                            'action': action,
                            'expert_reward': expert_reward_val,
                            'reward_label': reward_label
                        })
                
                # Restore environment state
                self.env.current_step = env_backup['current_step']
                self.env.capital = env_backup['capital']
                self.env.shares_held = env_backup['shares_held']
                self.env.current_position = env_backup['current_position']
                self.env.portfolio_values = env_backup['portfolio_values']
                self.env.returns = env_backup['returns']
                self.env.portfolio_peak = env_backup['portfolio_peak']
                self.env.max_drawdown = env_backup['max_drawdown']
                self.env.current_drawdown = env_backup['current_drawdown']
                self.env.done = False
            
            # Take a step with hold action to advance the environment (Gymnasium-style)
            state, _, terminated, truncated, _ = self.env.step(0)
            done = bool(terminated or truncated)
            
            if done:
                break
        
        print(f"Generated {len(samples)} expert-labeled samples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['state_sequence']),
            torch.LongTensor([sample['action']]),
            torch.FloatTensor([sample['expert_reward']]),
            sample['reward_label']
        )


def pretrain_reward_network(config, data, model_save_path):
    """
    Pre-train the reward network using supervised learning.
    
    Args:
        config (dict): Configuration dictionary
        data (pd.DataFrame): Training data
        model_save_path (str): Path to save the pre-trained model
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create dataset
    dataset = RewardPretrainingDataset(
        data=data,
        window_size=config['environment']['window_size'],
        reward_labels=config['training']['reward_labels'],
        expert_selection=config.get('training', {}).get('expert_selection', None)
    )
    
    # Create data loader with GPU optimizations
    dataloader = DataLoader(
        dataset,
        batch_size=config['pretraining']['batch_size'],
        shuffle=True,
        num_workers=4,          # Use multiple workers for faster data loading
        pin_memory=True,        # Pin memory for faster GPU transfer
        persistent_workers=True # Keep workers alive between epochs
    )
    
    # Calculate feature dimension (data features + position info)
    feature_dim = len(data.columns) + 1  # Data features + position info added by environment
    
    # Create reward network
    reward_network = RewardNetwork(
        state_dim=feature_dim,
        action_dim=3,  # Hold, Buy, Sell
        hidden_dim=config['model']['reward_net']['hidden_size'],
        seq_len=config['environment']['window_size'],
        model_type=config['model']['reward_net']['model_type'],
        num_layers=config['model']['reward_net'].get('num_layers', 2),
        dropout=config['model']['reward_net'].get('dropout', 0.1)
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        reward_network.parameters(),
        lr=config['pretraining']['learning_rate'],
        weight_decay=config['pretraining'].get('weight_decay', 1e-4)
    )
    
    # Mixed precision training for better GPU utilization
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    use_mixed_precision = config.get('mixed_precision', True) and device.type == 'cuda'
    
    if use_mixed_precision:
        logger.info("Using mixed precision training for faster GPU performance")
    
    # Training loop
    num_epochs = config['pretraining']['num_epochs']
    logger.info(f"Starting pre-training for {num_epochs} epochs...")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (state_sequences, actions, expert_rewards, reward_labels) in enumerate(progress_bar):
            # Move to device with non-blocking transfer for better GPU utilization
            state_sequences = state_sequences.to(device, non_blocking=True)
            actions = actions.squeeze(1).to(device, non_blocking=True)
            expert_rewards = expert_rewards.squeeze(1).to(device, non_blocking=True)
            
            # Mixed precision forward pass
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    predicted_rewards = reward_network(state_sequences, actions).squeeze()
                    
                    # Handle single sample case
                    if predicted_rewards.dim() == 0:
                        predicted_rewards = predicted_rewards.unsqueeze(0)
                    if expert_rewards.dim() == 0:
                        expert_rewards = expert_rewards.unsqueeze(0)
                    
                    # Compute loss
                    loss = F.mse_loss(predicted_rewards, expert_rewards)
            else:
                # Regular forward pass
                predicted_rewards = reward_network(state_sequences, actions).squeeze()
                
                # Handle single sample case
                if predicted_rewards.dim() == 0:
                    predicted_rewards = predicted_rewards.unsqueeze(0)
                if expert_rewards.dim() == 0:
                    expert_rewards = expert_rewards.unsqueeze(0)
                
                # Compute loss
                loss = F.mse_loss(predicted_rewards, expert_rewards)
            
            # Debug NaN issues
            if torch.isnan(loss):
                print(f"NaN loss detected!")
                print(f"Predicted rewards stats: min={predicted_rewards.min():.6f}, max={predicted_rewards.max():.6f}, mean={predicted_rewards.mean():.6f}")
                print(f"Expert rewards stats: min={expert_rewards.min():.6f}, max={expert_rewards.max():.6f}, mean={expert_rewards.mean():.6f}")
                print(f"Any NaN in predicted: {torch.any(torch.isnan(predicted_rewards))}")
                print(f"Any NaN in expert: {torch.any(torch.isnan(expert_rewards))}")
                break
            
            # Mixed precision backward pass
            optimizer.zero_grad()
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(reward_network.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reward_network.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': loss.item()})
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': reward_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': config
            }, model_save_path)
            logger.info(f"Saved best model with loss {avg_loss:.6f}")
    
    logger.info("Pre-training completed!")
    logger.info(f"Best model saved to {model_save_path} with loss {best_loss:.6f}")
    
    return reward_network


def main():
    """Main function for pre-training the reward network."""
    parser = argparse.ArgumentParser(description='Pre-train SRDDQN reward network')
    parser.add_argument('--config', type=str, default='configs/srddqn_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='data/processed/train_data.csv',
                        help='Path to training data')
    parser.add_argument('--output_path', type=str, default='models/saved/pretrained_reward_network.pth',
                        help='Path to save pre-trained model')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Add pretraining configuration if not present
    if 'pretraining' not in config:
        config['pretraining'] = {
            'num_epochs': 50,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        }
    
    # Load training data
    if os.path.exists(args.data_path):
        data = pd.read_csv(args.data_path)
        print(f"Loaded training data from {args.data_path}")
    else:
        # Generate data if not exists
        print("Training data not found, generating...")
        data_processor = DataProcessor(config['data'])
        data = data_processor.download_data(
            ticker=config['data']['ticker'],
            start_date=config['data']['train_start_date'],
            end_date=config['data']['train_end_date']
        )
        data = data_processor.add_technical_indicators(data)
        data = data_processor.normalize_data(data)
        
        # Save for future use
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        data.to_csv(args.data_path, index=False)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Pre-train the reward network
    reward_network = pretrain_reward_network(config, data, args.output_path)
    
    print(f"Pre-training completed! Model saved to {args.output_path}")


if __name__ == "__main__":
    main()