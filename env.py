import traci


# this code copied from RLMODEL.env3-onedrive 30/10/2024

class Env:
    def __init__(self):
        self.state = None
        self.rewards = {}
        self.action_state_pairs = {}
        self.step_count = 0

    def extract_state(self, junction_id):
        """

        extract the current state of the junction, include:
        + Road name/ID (edge)
        + queue lengths (number of vehicles per road)
        + Road lengths (for reward later)
        + current traffic light phase
        - total waiting time for vehicles (not needed now)

        * lanes: The raw list of lanes controlled by the traffic light, could contain duplicates (coming - going lanes).
        * unique_lanes: A cleaned up version of lanes, where duplicates are removed.
        """
        state = []
        # 1- Get the lanes controlled by the junction
        lanes = traci.trafficlight.getControlledLanes(junction_id)
        unique_lanes = list(set(lanes))  # get unique lanes without duplicates - incoming Traffic only -

        # 2- Get road lengths for all lanes controlled by the junction
        road_lengths = self.get_road_length(junction_id)
        # Lane lengths: {'-E1_0': 65.01, '-E2_0': 43.56, 'E0_0': 66.65, '-E3_0': 39.7}

        # 3- get queue lengths for all lanes controlled by the junction
        queue_lengths = self.get_queue_length(junction_id)
        # Queue lengths: {'-E1_0': 0, '-E2_0': 1, 'E0_0': 0, '-E3_0': 0}

        # add lane IDs, road lengths, and queue lengths to the state
        for lane in unique_lanes:
            lane_length = road_lengths.get(lane, 0)  # Default to 0 if no length is found
            queue_length = queue_lengths.get(lane, 0)  # Default to 0 if no vehicles are found

            # Add lane ID, length, and queue length to the state
            state.append((lane, lane_length, queue_length))

        # 4- Get current traffic light phase
        current_phase = self.get_current_phase(junction_id)
        state.append(current_phase)  # add current traffic light phase to the state

        return tuple(state)

    def take_action(self, action, junction_id):
        """
        Prioritize lanes with vehicles and set the green light for the lane with the highest queue length
        Lanes without vehicles remain red
        """
        #   phases to descriptions ---- not needed now -----
        phases = {
            0: "Phase 0: West-East Green",
            1: "Phase 3: South-North Green",
            2: "Phase 6: East-West Green",
            3: "Phase 9: North-South Green"
        }

        # extract the state, including lane info (queue lengths and road lengths)
        state = self.extract_state(junction_id)

        # extract lane information from the state (ignoring the last element - traffic light phase)
        lanes_info = state[:-1]

        # organize lanes based on queue lengths (3rd element in each tuple) in descending order
        sorted_lanes = sorted(lanes_info, key=lambda x: x[2], reverse=True)  # Sort by queue length (x[2])

        # Determine the phase to activate
        if all(lane[2] == 0 for lane in sorted_lanes):
            # If no vehicles are detected keep the current phase
            print("No vehicles detected, keeping Traffic Light Logic as normal")
            action_phase = traci.trafficlight.getPhase(junction_id)  # Keep the current phase
        else:
            # Prioritize the incoming lane with the highest queue length
            highest_queue_lane = sorted_lanes[0]  # Lane with the highest queue length
            highest_lane_id = highest_queue_lane[0]  # The lane ID with the highest queue length
            # ("E0_0", 66.65, 3)

            print(f"Incoming lane {highest_lane_id} has the highest queue length: {highest_queue_lane[2]} Setting it green")
            # Queue lengths: {'-E1_0': 0, '-E2_0': 3, 'E0_0': 0, '-E3_0': 0}
            # Incoming lane -E2_0 has the highest queue length: 3. Setting it green

            # map incoming lane IDs to the corresponding traffic light phases
            lane_to_phase = {
                "E0_0": 0,  # West-East (incoming traffic)
                "-E1_0": 6,  # East-West (incoming traffic)
                "-E2_0": 3,  # South-North (incoming traffic)
                "-E3_0": 9  # North-South (incoming traffic)
            }

            # determine the phase to activate for the lane with the highest queue
            if highest_lane_id in lane_to_phase:
                action_phase = lane_to_phase[highest_lane_id]
            else:
                print(f"Lane {highest_lane_id} is not an incoming lane. Keeping the current phase.")
                action_phase = traci.trafficlight.getPhase(junction_id)  # Keep the current phase if no match is found

        # get the current state before action
        current_state = self.state

        # execute the phase change in the execute_TL_action function
        new_state = self.execute_TL_action(junction_id, action_phase)

        # calculate reward based on traffic change
        reward = self.calculate_reward(current_state, new_state)

        # Update - register the rewards table (for tracking)
        if current_state not in self.rewards:
            self.rewards[current_state] = {}
        if action not in self.rewards[current_state]:
            self.rewards[current_state][action] = 0
        self.rewards[current_state][action] += reward

        # update state and check if done
        self.state = new_state
        done_flag = self.check_done_condition(junction_id)

        # Return the new state, reward, current state, done flag, and reward
        return new_state, reward, current_state, done_flag, reward

    def execute_TL_action(self, junction_id, phase):
        """
        Execute the traffic light phase change at the given junction - lane
        """
        # Set the traffic light phase to the decided action
        print(f"Setting phase {phase} at junction {junction_id}.")
        traci.trafficlight.setPhase(junction_id, phase)

        # Advance the simulation for a few steps
        traci.simulationStep()

        # Get the new state after executing the action
        new_state = self.extract_state(junction_id)
        return new_state

    def calculate_reward(self, previous_state, current_state):
        """
        Reward is based on minimizing the queue length of the lane ---- not working well need to check agine why
        Compare queue lengths from the previous and current states to determine the reward
        """
        # extract the queue lengths from previous and current states
        previous_queue_lengths = [lane[2] for lane in previous_state[:-1]]  # Extract queue lengths (3rd element)
        current_queue_lengths = [lane[2] for lane in current_state[:-1]]  # Extract queue lengths (3rd element)

        # Sum the queue lengths for both previous and current states
        total_previous_queue_length = sum(previous_queue_lengths)
        total_current_queue_length = sum(current_queue_lengths)

        # calculate the reward based on the reduction in queue lengths
        reward = total_previous_queue_length - total_current_queue_length  # Positive reward if queues decrease

        print(f"Previous Queue Length: {total_previous_queue_length}, Current Queue Length: {total_current_queue_length}, Reward: {reward}")

        return reward

    def check_done_condition(self, junction_id):
        """
        Check if the episode is done based on:
        - Reaching a step count limit
        - All lanes having no vehicles in the queue
        """
        # add the step count
        self.step_count += 1

        # check if the step count limit is reached
        if self.step_count >= 100:
            return True

        # Check if all lanes are cleared (no vehicles in the queue)
        queue_lengths = self.get_queue_length(junction_id)

        # if all queue lengths are 0, end the episode
        if all(queue_length == 0 for queue_length in queue_lengths.values()):
            print("All lanes are clear, ending the episode early.")
            return True

        return False

    def get_road_length(self, junction_id):
        """
        This function calculates the length of each lane controlled by the junction.
        :param junction_id: The ID of the traffic light junction
        :return: A dictionary with lane IDs as keys and their lengths as values
        """
        lanes = traci.trafficlight.getControlledLanes(junction_id)
        unique_lanes = list(set(lanes))  # Get unique lanes

        lane_lengths = {}
        for lane in unique_lanes:
            length = traci.lane.getLength(lane)  # Get the length of the lane
            lane_lengths[lane] = length

        # Print the road lengths for debugging purposes
        print(f"Lane lengths: {lane_lengths}")

        return lane_lengths

    def get_queue_length(self, junction_id):
        """
        This function calculates the queue length (number of vehicles) for each lane controlled by the junction.
        :param junction_id: The ID of the traffic light junction
        :return: A dictionary with lane IDs as keys and their queue lengths as values
        """
        lanes = traci.trafficlight.getControlledLanes(junction_id)
        unique_lanes = list(set(lanes))  # Get unique lanes

        queue_lengths = {}
        for lane in unique_lanes:
            queue_length = traci.lane.getLastStepVehicleNumber(lane)  # Get the queue length for each lane
            queue_lengths[lane] = queue_length

        # Print the queue lengths for debugging purposes
        print(f"Queue lengths: {queue_lengths}")

        return queue_lengths

    def get_current_phase(self, junction_id):
        """
        Get the current traffic light phase at the junction.
        :param junction_id: The ID of the traffic light junction
        :return: The current phase index.
        """
        current_phase = traci.trafficlight.getPhase(junction_id)
        return current_phase

    def reset(self, junction_id):
        """
        reset the environment to its init state at the start of each episode
        This includes resetting the state, clearing action-state pairs, and resetting step count
        """
        self.state = self.extract_state(junction_id)  # Extract the initial state
        self.rewards = {}  # Clear rewards tracking
        self.action_state_pairs = {}  # Clear action-state pairs tracking
        self.step_count = 0  # Reset step counter

        print(f"Environment reset for junction {junction_id}. Initial state: {self.state}")

        return self.state  # return the initial state to start the episode
