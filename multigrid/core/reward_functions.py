import numpy as np
import copy
from math import log2
from collections import defaultdict



# Helper function to compute Shannon entropy
def shannon_entropy(prob_dist):
    entropy = 0
    for p in prob_dist:
        if p > 0:
            entropy -= p * log2(p)
    return entropy

# Discrete Choice Function
def discrete_choice(transition_probs, current_state, n_steps):
    reachable_states = set()
    
    # Initialize with current state
    current_states = [(current_state, 1)]
    
    for _ in range(n_steps):
        next_states = defaultdict(float)
        
        for state, prob in current_states:
            for next_state, transition_prob in transition_probs[state].items():
                next_states[next_state] += prob * transition_prob
                
        current_states = list(next_states.items())
        
        for state, _ in current_states:
            reachable_states.add(state)
            
    return len(reachable_states)


# Modified function for a multi-agent environment, focusing only on agent '0'
def estimate_entropic_choice_multi_agent(env, policies, n_steps=3, n_samples=100):
    """
    Estimate the entropic choice for a given environment and policies for multiple agents using Monte Carlo sampling.
    Only the entropic choice for agent '0' is computed.
    
    Parameters:
        env: The environment object, assumed to be OpenAI Gym-like with multi-agent support.
        policies: A dictionary of policy functions, each taking a state and returning an action, keyed by agent IDs.
        n_steps: The number of steps to look ahead.
        n_samples: The number of Monte Carlo samples to use for the estimate.
    
    Returns:
        entropic_choice_0: The estimated entropic choice for agent '0'.
    """
    # Initialize variables to keep track of state visit frequencies for agent '0'
    state_frequencies_0 = {}
    
    # Save the initial state of the environment
    env_copy = copy.deepcopy(env)
    
    for _ in range(n_samples):
        # Reset to the saved initial state
        env = copy.deepcopy(env_copy)
        
        # Rollout for n_steps to collect states
        states = env.reset()
        for _ in range(n_steps):
            actions = {agent_id: policy(states[agent_id]) for agent_id, policy in policies.items()}
            next_states, _, _, _ = env.step(actions)
            
            # Increment state visit frequency for agent '0'
            state_0 = tuple(next_states['0'])
            if state_0 in state_frequencies_0:
                state_frequencies_0[state_0] += 1
            else:
                state_frequencies_0[state_0] = 1
            
            states = next_states
    
    # Calculate entropic choice for agent '0'
    total_samples_0 = sum(state_frequencies_0.values())
    probabilities_0 = np.array(list(state_frequencies_0.values())) / total_samples_0
    entropic_choice_0 = -np.sum(probabilities_0 * np.log(probabilities_0))
    
    return entropic_choice_0




# Immediate Choice Function
def immediate_choice(transition_probs, current_state):
    next_state_probs = list(transition_probs[current_state].values())
    return shannon_entropy(next_state_probs)

# Sample transition probabilities for demonstration (state -> next_state -> probability)
# In the format {current_state: {next_state1: prob1, next_state2: prob2, ...}, ...}
sample_transition_probs = {
    'A': {'A': 0.1, 'B': 0.4, 'C': 0.5},
    'B': {'A': 0.3, 'B': 0.3, 'C': 0.4},
    'C': {'A': 0.2, 'B': 0.2, 'C': 0.6}
}

# Test the functions
current_state = 'A'
n_steps = 2

dc_result = discrete_choice(sample_transition_probs, current_state, n_steps)
ec_result = entropic_choice(sample_transition_probs, current_state, n_steps)
ic_result = immediate_choice(sample_transition_probs, current_state)

dc_result, ec_result, ic_result


