import traci
from RL_agent import RL
from env import Env
import time

class RLRunnerSUMO:
    def __init__(self, epsilon_point, RL_Agent, algo_name, junction_id, episodes=1000):
        # Init RL agent hyperparameters
        self.epsilon = 1.0  # High means more exploration; low means exploiting the action with the highest Q-value
        self.learning_rate = 0.7  # Learning rate
        self.gamma = 0.9  # High long-term reward; low immediate reward (short term)
        self.epsilon_decay_rate = 0.005  # Decay rate for epsilon
        self.min_epsilon = 0.05
        self.epsilon_point = epsilon_point
        self.algo_name = algo_name
        self.junction_id = junction_id
        self.episodes = episodes  # Num of episodes passed in init

        # Init RL agent
        self.rl_agent = RL_Agent(self.epsilon, self.learning_rate, self.epsilon_decay_rate, self.min_epsilon, self.epsilon_point)
        self.env = Env()  # Init SUMO environment

        # metric for tracking performance later for plot
        self.total_rewards = []
        self.total_steps_per_episode = []

    def run(self, num_iterations=2000):
        """
        Runs the Q-learning for number of episodes for multiple iterations
        """
        for episode in range(self.episodes):
            print(f"Episode {episode + 1}/{self.episodes}")
            # Reset the environment and get the init state
            self.env.reset(self.junction_id)
            total_episode_reward = 0  # track total reward for this episode
            steps_in_episode = 0  # reset steps for the episode

            for iteration in range(num_iterations):
                print(f"Iteration {iteration + 1}/{num_iterations}")
                done = False

                while not done:
                    state = self.env.state  # get the current state

                    # choose an action using the RL agent's policy (explore or exploit)
                    action = self.rl_agent.action_policy(state, [0, 1, 2, 3])  # Available actions: 0, 1, 2, 3 (phases)

                    # execute the action in the environment and observe the next state and reward
                    new_state, reward, current_state, done_flag, _ = self.env.take_action(action, self.junction_id)

                    # update Q-values using Q-learning update rule
                    self.rl_agent.update_q_values(current_state, action, reward, new_state, gamma=self.gamma)

                    # update metrics and environment state
                    total_episode_reward += reward  # accumulate reward
                    self.env.state = new_state  # move to the next state
                    steps_in_episode += 1  # add the step count in this episode

                    # advance the simulation in SUMO
                    traci.simulationStep()

                    # check if the episode is done (depend on traffic in the simulation or steps)
                    done = done_flag or self.env.check_done_condition(self.junction_id)

            # store episode performance metrics after all iterations for this episode
            self.total_rewards.append(total_episode_reward)
            self.total_steps_per_episode.append(steps_in_episode)

            # decay epsilon after each episode
            self.rl_agent.decay_epsilon(episode)

            print(f"Episode {episode + 1} completed with total reward: {total_episode_reward} and {iteration} steps")

        # end SUMO simulation after all episodes
        traci.close()

    def display_results(self):
        """
        Display the results of the Q-learning after training is completed --- need to make sure
        """
        print(f"Training completed over {self.episodes} episodes.")
        print(f"Total rewards per episode: {self.total_rewards}")
        print(f"Steps per episode: {self.total_steps_per_episode}")
        print("Final Q-values after training:")

        # display the learned Q-values
        for state, actions in self.rl_agent.q_values.items():
            print(f"State: {state}, Actions: {actions}")


if __name__ == "__main__":
    # SUMO command to start the simulation
    sumo_cmd = ['sumo-gui', '-c', 'C:/Users/psmalz12/OneDrive/PGR/pycharm/RLMODEL2/sumo.sumocfg']

    # Start SUMO simulation
    traci.start(sumo_cmd)

    # Init and run the RL agent in the SUMO environment
    epsilon_point = 50  # where epsilon reaches its minimum value
    rl_runner_sumo = RLRunnerSUMO(epsilon_point, RL, "Q-Learning", "J1", episodes=100)

    # run the training process passing the number of iterations
    rl_runner_sumo.run(num_iterations=500)

    # display the results after training is completed
    rl_runner_sumo.display_results()
