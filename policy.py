import random


env_grid = "5x5"  # Or "10x10"


if env_grid == "5x5":
    from env5 import Grid
elif env_grid == "10x10":
    from env10 import Grid2 as Grid
elif env_grid == "50x50":
    from env50 import Grid3 as Grid
elif env_grid == "20x20":
    from env20 import Grid4 as Grid
else:
    print("Cannot import the environment")

# optimal policy from Q-values
def optimal_policy(q_values):
    optimal_policy = {}

    for state, actions in q_values.items():
        # Find the highest Q-value for the current state
        max_q_value = max(actions.values())

        # Get all actions that have the highest q-value (dic)
        best_actions = [action for action, q_value in actions.items() if q_value == max_q_value]

        # choose randomly one of the actions if there are more than 1
        optimal_action = random.choice(best_actions)
        optimal_policy[state] = optimal_action

    return optimal_policy

# optimal path from current state to the goal state
def find_optimal_path(current_state, optimal_policy, grid):
    # Access the goal state from the grid object
    goal_state = grid.goal
    optimal_path = [current_state]

    while current_state != goal_state:
        action = optimal_policy[current_state]
        new_state, _,_, _,_ = grid.take_action(action)  # Ignore other return values
        optimal_path.append(new_state)
        current_state = new_state

    return optimal_path
