def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    new_values = []
    
    for s in range(num_states):
        best = float('-inf')  # best Q value over actions
        
        # iterate over actions available in state s
        for a in range(len(transitions[s])):
            
            # compute expected value for this action
            q = rewards[s][a]
            
            expected = 0.0
            for s_next in range(num_states):
                expected += transitions[s][a][s_next] * values[s_next]
            
            q += gamma * expected
            
            # take max over actions
            best = max(best, q)
        
        new_values.append(best)
    
    return new_values