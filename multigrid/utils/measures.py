import numpy as np 

def immediate_choice(self, policy, state):
    """
    Compute the immediate choice for a given policy and state for the given agent.

    Parameters
    ----------
    policy : function
        The policy function. We are using PPO in this instance (subject to change). This takes a state as input and returns a probability distribution over possible actions to take and their associated reward.
    state : any
        The current state of the environment.

    Returns
    -------
    entropy : 
    """
    # Get the probability distribution over actions for the given state
    action_probs = policy.predict(state)[0]

    # Compute the entropy of the action probabilities
    entropy = -np.sum(action_probs * np.log(action_probs))

    return entropy

def entropic_choice(self, policy, state, n):
    continue
