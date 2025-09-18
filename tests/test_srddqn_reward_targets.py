import numpy as np
import torch
import pytest
from src.models.agents.srddqn import SRDDQNAgent
from src.models.agents.double_dqn import DoubleDQNAgent


def test_per_action_target_construction(monkeypatch):
    # Minimal dims
    state_dim = 4  # flattened observation size
    action_dim = 3
    seq_len = 2

    # Create SRDDQNAgent with CPU device for test
    agent = SRDDQNAgent(state_dim=state_dim, action_dim=action_dim, seq_len=seq_len, device='cpu')
    # Use small batch size so train() runs with our single synthetic entry
    agent.batch_size = 1
    agent.dqn_agent.batch_size = 1

    # Create a synthetic orig_state sequence (seq_len x reward_input_dim)
    # reward_input_dim = window_size * feature_dim; with our simplistic inputs we will craft accordingly
    reward_input_dim = agent.reward_input_dim
    orig_seq = np.zeros((seq_len, reward_input_dim), dtype=np.float32)
    orig_seq[0] = np.array([1.0] * reward_input_dim, dtype=np.float32)
    orig_seq[1] = np.array([2.0] * reward_input_dim, dtype=np.float32)

    # Expert reward dict: choose a label and known value
    expert_label = agent.reward_labels[0]
    expert_value = 0.5
    expert_dict = {expert_label: expert_value}

    # Predicted self rewards per action (action_dim length)
    preds = np.array([0.2, 0.8, -0.1], dtype=np.float32)

    # Stored scalar reward in the buffer (simulate final_reward for the sampled action)
    # We'll place a positive stored reward indicating the chosen action's final_reward
    stored_reward = 1.23

    # Build a fake flattened state to match buffer expectation
    flat_state = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    flat_next_state = flat_state.copy()

    # Insert synthetic entry into the DoubleDQNAgent's replay buffer directly
    # The ReplayBuffer.add signature accepts predicted_self_rewards, orig_state, expert_reward_dict
    dqn = agent.dqn_agent
    # Ensure buffer arrays get initialized
    dqn.replay_buffer.add(flat_state, 0, 0.0, flat_next_state, False)

    # Now overwrite the last inserted slot with our synthesized extended data
    idx = (dqn.replay_buffer.position - 1) % dqn.replay_buffer.capacity
    # Place predicted_self_rewards into the list/array storage
    try:
        # If contiguous array exists, write into it
        if dqn.replay_buffer._predicted_self_rewards_arr is not None:
            dqn.replay_buffer._predicted_self_rewards_arr[idx] = preds
        else:
            dqn.replay_buffer._predicted_self_rewards[idx] = preds
    except Exception:
        dqn.replay_buffer._predicted_self_rewards[idx] = preds

    # Put orig_state
    try:
        if dqn.replay_buffer._orig_states_arr is not None:
            dqn.replay_buffer._orig_states_arr[idx] = orig_seq
        else:
            dqn.replay_buffer._orig_states[idx] = orig_seq
    except Exception:
        dqn.replay_buffer._orig_states[idx] = orig_seq

    # Put expert dict
    dqn.replay_buffer._expert_reward_dicts[idx] = expert_dict

    # Put stored scalar reward in scalar rewards array
    dqn.replay_buffer._rewards[idx] = float(stored_reward)

    # For controlled behavior, patch SRDDQNAgent.train_reward_network_batch to capture experiences
    captured = {}

    def fake_train_reward_network_batch(experiences):
        # Save experiences for assertion
        captured['experiences'] = experiences
        return 0.0

    monkeypatch.setattr(agent, 'train_reward_network_batch', fake_train_reward_network_batch)

    # Force scheduling conditions: set steps_done >= warmup and set steps_done to multiple of update interval
    agent.steps_done = agent.reward_warmup_steps
    agent.reward_update_interval = 1  # train every call
    agent.episode_reward_train_count = 0

    # Call train() — this will call dqn_agent.train() then build per-action experiences and call our fake trainer
    dqn_loss, reward_loss = agent.train()

    # Verify that experiences were captured
    assert 'experiences' in captured, "train_reward_network_batch was not called"
    exps = captured['experiences']

    # Expectation: since predicted_self_rewards exist, for each action there should be an example.
    assert len(exps) == action_dim, f"Expected {action_dim} per-action examples, got {len(exps)}"

    # Map action->target expected:
    expected_targets = []
    for act_idx in range(action_dim):
        # If act == sampled action (we used action 0 when adding) -> prefer stored_reward
        if act_idx == 0:
            expected_targets.append(float(stored_reward))
        else:
            expected_targets.append(float(max(float(preds[act_idx]), float(expert_value))))

    # Extract targets from captured experiences and compare
    actual_targets = [float(t[2]) for t in exps]

    assert actual_targets == pytest.approx(expected_targets), f"Targets mismatch. expected={expected_targets}, actual={actual_targets}"


if __name__ == '__main__':
    test_per_action_target_construction(lambda *a, **k: None)
    print('ok')
