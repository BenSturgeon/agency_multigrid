import numpy as np
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

# Entropic Choice Function
def entropic_choice(transition_probs, current_state, n_steps):
    state_probs = defaultdict(float)
    
    # Initialize with current state
    current_states = [(current_state, 1)]
    
    for _ in range(n_steps):
        next_states = defaultdict(float)
        
        for state, prob in current_states:
            for next_state, transition_prob in transition_probs[state].items():
                next_states[next_state] += prob * transition_prob
                
        current_states = list(next_states.items())
        
    state_probs = [prob for _, prob in current_states]
    
    return shannon_entropy(state_probs)

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
