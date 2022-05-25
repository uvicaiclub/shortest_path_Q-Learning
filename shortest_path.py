import numpy as np

# Define the states
location_to_state = {
    'L1': 0,
    'L2': 1,
    'L3': 2,
    'L4': 3,
    'L5': 4,
    'L6': 5,
    'L7': 6,
    'L8': 7,
    'L9': 8
}

# Define the actions
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Define the rewards
rewards = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 1, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 0]])

# Create a dictionary mapping from state to location
state_to_location = dict((state, location) for location, state in location_to_state.items())

# Initialize model parameters
gamma = 0.75  # Discount rate
alpha = 0.9  # learning rate


# Create Q-Learning agent class
class QAgent():

    # Initialize attributes
    def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_to_location):
        self.gamma = gamma
        self.alpha = alpha
        self.location_to_state = location_to_state
        self.actions = actions
        self.rewards = rewards
        self.state_to_location = state_to_location

        # create an NxN Q-table initialized with all zeros
        N = len(location_to_state)
        self.Q = np.zeros((N, N), dtype=int, order='C')

    # Train the agent in the environment
    def train(self, start_location, end_location, iterations):
        rewards_new = np.copy(self.rewards)
        ending_state = self.location_to_state[end_location]
        # give end_location a large reward
        rewards_new[ending_state, ending_state] = 100

        # pick a random current state
        for i in range(iterations):
            current_state = np.random.randint(0, 9)
            playable_actions = []

            '''
            Iterate through the rewards matrix to get the states which
            are directly reachable from the randomly chosen current state.
            Then add those states to a list called playable_actions.
            '''
            for j in range(9):
                if rewards_new[current_state, j] > 0:
                    playable_actions.append(j)

            # choose the next random state
            next_state = np.random.choice(playable_actions)

            # Bellman's Equation (finding temporal difference)
            TD = rewards_new[current_state, next_state] \
                 + self.gamma * self.Q[next_state, np.argmax(self.Q[next_state,])] \
                 - self.Q[current_state, next_state]

            # Update the Q-table with new Q-values
            self.Q[current_state, next_state] += self.alpha * TD
            '''
            if (i+1) % 100 == 0:
                print('================= Iteration {} ================='.format(i))
                print('Q-Table:')
                print(self.Q)

                self.get_optimal_route(start_location, end_location, self.Q)
            '''

        # Get the optimized route
        self.get_optimal_route(start_location, end_location, self.Q)

    # get the optimal route
    def get_optimal_route(self, start_location, end_location, Q):
        # initialize the route and next_location
        route = [start_location]
        next_location = start_location

        # append the next location to the route list then print that list
        while next_location != end_location:
            starting_state = self.location_to_state[start_location]
            next_state = np.argmax(Q[starting_state,])
            next_location = self.state_to_location[next_state]
            route.append(next_location)
            start_location = next_location

        print(route)


# initialize and train an agent
qagent = QAgent(alpha, gamma, location_to_state, actions, rewards, state_to_location)
# train the qagent to find the optimal route from L1 to L9 over 1000 iterations
qagent.train('L1', 'L9', 1000)
