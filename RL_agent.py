import random
# q-leartning
# In Q-learning the agent does not know the specific action that will be taken in the next state
# it updates the Q-value using the highest Q-value among all possible actions in that state.
# By assuming the best possible action will be chosen, this approach results in an optimistic update.
# This optimism helps the agent gradually learn the optimal policy over time

class RL:
    def __init__(self, epsilon, learning_rate, epsilon_decay_rate, min_epsilon, epsilon_point):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q_values = {} # q value table
        self.st_epsilon = epsilon  # Store the init value of epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_point = epsilon_point
        self.min_epsilon = min_epsilon



    def action_policy(self, state, available_actions):
        if random.random() < self.epsilon:
            # explor: choose a random action
            return random.choice(available_actions)
        else:
            # exploit: choose the action with the highest q-value
            if state in self.q_values:
                return max(self.q_values[state], key=self.q_values[state].get) # the key res of the max based on the q-values
            else:
                # if state is not in Q-values, choose a random action
                return random.choice(available_actions)



    def update_q_values(self, state, action, reward, next_state, gamma):
        # init q-value if not present in the q-value table
        if state not in self.q_values:
            self.q_values[state] = {}
        if action not in self.q_values[state]:
            self.q_values[state][action] = 0

        # Update q-value using the standard Q-learning update rule with gamma
        self.q_values[state][action] += self.learning_rate * (reward + gamma * max(self.q_values.get(next_state, {}).values(), default=0) - self.q_values[state][action])
        # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
        # the Q-value update uses the maximum possible Q-value of the next state regardless of the action the agent actually takes.
        # this means that Q-Learning aims to learn the optimal policy directly by considering the best action in the next state
        # Return the updated q-value
        return self.q_values[state][action]

    def decay_epsilon(self, episode):
        if episode < self.epsilon_point:
            # decayed epsilon based on episode number
            self.epsilon = max(0,self.st_epsilon - self.epsilon_decay_rate * episode / self.epsilon_point * self.st_epsilon)
        else:
            # when reaching the epsilon_point episode remains at min_epsilon
            self.epsilon = self.min_epsilon

    def reset_epsilon(self):
        # Reset epsilon to init value
        self.epsilon = self.st_epsilon