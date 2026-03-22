import numpy as np
import copy
import pprint
import pandas as pd

TILE_REWARD = {
    "White": -0.05,
    "Brown": -1,
    "Green": 1,
}

ACTIONS = {
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3
}


class GridWorld:

    def __init__(self, tile_reward= TILE_REWARD, map = None, size = None):
        self.tile_reward = tile_reward
        if map == None:
            self.map = [["Green", "Wall", "Green", "White", "White", "Green"],
                    ["White", "Brown", "White", "Green","Wall","Brown"],
                    ["White","White", "Brown", "White", "Green","White"],
                    ["White", "White", "White", "Brown", "White", "Green"],
                    ["White", "Wall","Wall","Wall","Brown","White"],
                    ["White","White","White","White","White","White"]]
        else:
            self.map = map
        if size == None:
            self.values_map = np.zeros((6,6))
        else:
            self.values_map = np.zeros((size, size))

        self.states = [(c, r) for r, row in enumerate(self.map) for c, tile in enumerate(row) if tile != "Wall"]

    def step(self, pos, action):
        x_, y_ = copy.deepcopy(pos)

        if self.is_valid_action(x_, y_, action) is False:
            return pos, self.get_reward(pos)

        if action == ACTIONS["UP"]:
            y_ += -1
        elif action == ACTIONS["DOWN"]:
            y_ += 1
        elif action == ACTIONS["LEFT"]:
            x_ += -1
        elif action == ACTIONS["RIGHT"]:
            x_ += 1

        reward = self.get_reward((x_, y_))
        next_pos = (x_, y_)

        return next_pos, reward

    def get_reward(self, pos: tuple):
        x, y = pos
        # print(pos)
        return self.tile_reward[self.map[y][x]]

    def is_valid_action(self, x, y, action):

        x_, y_ = x, y

        if action == ACTIONS["UP"]:
            y_ += -1
        elif action == ACTIONS["DOWN"]:
            y_ += 1
        elif action == ACTIONS["LEFT"]:
            x_ += -1
        elif action == ACTIONS["RIGHT"]:
            x_ += 1

        if len(self.map) <= y_ or y_ < 0:
            return False
        elif len(self.map[0]) <= x_ or x_ < 0:
            return False
        elif self.map[y_][x_] == "Wall":
            return False

        return True


class Agent():
    def __init__(self, env, gamma, policy={}, actions=ACTIONS):
        self.actions = actions
        self.gamma = gamma
        if not policy:
            for s in env.states:
                policy[s] = None
        self.policy = policy
        self.env = env
        self.value_history ={}
        for s in env.states:
            self.value_history[s]=[]
        self.v = self.env.values_map

    def get_action_probs(self, action):
        if action == 0:
            return [0, 2, 3], [0.8, 0.1, 0.1]
        if action == 1:
            return [1, 2, 3], [0.8, 0.1, 0.1]
        if action == 2:
            return [2, 0, 1], [0.8, 0.1, 0.1]
        if action == 3:
            return [3, 0, 1], [0.8, 0.1, 0.1]

    def value_iteration(self):
        theta = 0.00101
        count = 0
        all_states = self.policy.keys()
        flag = True

        while flag:
            count += 1
            #if count == 36:  #specifies when to break iteration if we do not use theta as nreak condition
                #break
            print(count)
            delta = 0
            v_tmp = copy.deepcopy(self.v)
            for s in all_states:
                x, y = s
                max_a_value = -1

                for action in self.actions.values():
                    a_value = 0
                    actionl, probs = self.get_action_probs(action)
                    for a, p in zip(actionl, probs):
                        s_, r = self.env.step(s, a)
                        x_, y_ = s_                                     #we put r inside as p sums to 1 anyway
                        a_value += p * (r + self.gamma * v_tmp[y_][x_]) #after iteration ends

                    if a_value > max_a_value:

                        max_a_value = a_value
                        argmax_action = action
                        self.policy[s] = argmax_action
                        self.v[y][x] = max_a_value
                        delta1= abs(v_tmp[y][x] - self.v[y][x])

                self.value_history[(x, y)].append(self.v[y][x])

                delta = max(delta, delta1)
                if delta < theta:
                    flag = False
                    #print(delta, x,y)



if __name__ == '__main__':

    def generate_map(n):
        # Define the colors and wall as strings
        elements = ["Brown", "White", "Wall", "Green"]

        # Generate a NxN 2D array with random selection of the elements
        map_nxn = np.random.choice(elements, size=(n, n), p=[0.25, 0.5, 0.15, 0.1])

        map_list = map_nxn.tolist()
        return map_list

    bonus = generate_map(3)
    env = GridWorld(map=bonus, size=3)
    gamma = 0.99
    is_stable = False
    count = 0

    agent = Agent(env, gamma)
    agent.value_iteration()
        # print(count)


    print("Values for each state:")
    print(agent.v)
    print()
    print("Agent policy:")
    pprint.pprint((agent.policy))

    df_dict = {str(key): value for key, value in agent.value_history.items()}
    df = pd.DataFrame.from_dict(df_dict)
    df.loc[0] = 0
    df.to_csv(r"C:\Users\estee\Desktop\SC4003-Intelligent-Agents\assignment_1\bonus_value_iteration.csv", index=False)