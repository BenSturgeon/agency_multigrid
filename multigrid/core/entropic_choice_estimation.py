# Importing required modules
import numpy as np
import copy

def estimate_entropic_choice(env, policy, n_steps=3, n_samples=100):
    """
    Estimate the entropic choice for a given environment and policy using Monte Carlo sampling.
    
    Parameters:
        env: The environment object, assumed to be OpenAI Gym-like.
        policy: The policy function, which takes a state and returns an action.
        n_steps: The number of steps to look ahead.
        n_samples: The number of Monte Carlo samples to use for the estimate.
    
    Returns:
        entropic_choice: The estimated entropic choice.
    """
    # Initialize variables to keep track of state visit frequencies
    state_frequencies = {}
    
    # Save the initial state of the environment
    env_copy = copy.deepcopy(env)
    
    for _ in range(n_samples):
        # Reset to the saved initial state
        env = copy.deepcopy(env_copy)
        
        # Rollout for n_steps to collect states
        state = env.reset()
        for _ in range(n_steps):
            action = policy(state)
            next_state, _, _, _, _ = env.step(action)
            
            # Increment state visit frequency
            if tuple(next_state) in state_frequencies:
                state_frequencies[tuple(next_state)] += 1
            else:
                state_frequencies[tuple(next_state)] = 1
            
            state = next_state
    
    env = copy.deepcopy(env_copy)
    # Calculate entropic choice
    total_samples = sum(state_frequencies.values())
    probabilities = np.array(list(state_frequencies.values())) / total_samples
    entropic_choice = -np.sum(probabilities * np.log(probabilities))
    
    return entropic_choice

# Example usage (assuming `your_policy_function` and `your_environment` are defined)
# entropic_choice_value = estimate_entropic_choice(your_environment, your_policy_function)

