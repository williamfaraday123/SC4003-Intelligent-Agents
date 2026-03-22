import numpy as np
import copy
import pandas as pd
import pprint



TILE_REWARD = {
    "White": -0.05,
    "Brown":-1,
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
        x, y = copy.deepcopy(pos)
        x_, y_ = x,y

        if self.is_valid_action(x, y, action) is False:
            return pos, self.get_reward(pos)

        if action == ACTIONS["UP"]:
            y_ = y-1
        elif action == ACTIONS["DOWN"]:
            y_ = y+1
        elif action == ACTIONS["LEFT"]:
            x_ = x-1
        elif action == ACTIONS["RIGHT"]:
            x_ = x+1

        reward = self.get_reward((x, y))
        next_pos = (x_, y_)

        return next_pos, reward
    
    def get_reward(self, pos: tuple):
        x, y = pos
        #print(pos)
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

        if len(self.map) <= y_  or y_ < 0:
            return False
        elif len(self.map[0]) <= x_ or x_ < 0:
            return False
        elif self.map[y_][x_] == "Wall":
            return False

        return True


class Agent():
    def __init__(self, env, gamma, policy = {}, actions= ACTIONS):
        self.actions = actions
        self.gamma = gamma
        if not policy:
            for s in env.states:
                policy[s] = 0
        self.policy = policy
        self.value_history ={}
        for s in env.states:
            self.value_history[s]=[]

        self.env = env
        self.v = self.env.values_map

        
    def get_action_probs(self, action):
        if action == 0:
            return [0,2,3], [0.8,0.1,0.1]
        if action == 1:
            return [1,2,3], [0.8,0.1,0.1]
        if action == 2:
            return [2,0,1], [0.8,0.1,0.1]
        if action == 3:
            return [3,0,1], [0.8,0.1,0.1]

    def evaluate_policy(self):
        theta = .001
        count = 0
        all_states = self.policy.keys()
        delta = np.inf
        while delta >= theta:
            count+=1
            #print(count)
            delta = 0
            v_tmp = copy.deepcopy(self.v)
            for s in all_states:
                x, y = s

                actions, probs = self.get_action_probs(self.policy[s])
                value = 0
                for a, p in zip(actions,probs):
                    s_, r = self.env.step(s, a)
                    x_, y_ = s_
                    value += p*(r + self.gamma * v_tmp[y_][x_]) #we can put r inside as p sums to 1 anyway

                self.v[y][x] = value
                #self.value_history[(x,y)].append(value) #uncomment this if we want to record value history at every
                                                         #policy evaluation step
                delta = max(delta, abs(v_tmp[y][x] - self.v[y][x]))

            if delta <= theta:
                for s in all_states:
                    x, y = s
                    self.value_history[(x, y)].append(self.v[y][x])



    def improve_policy(self):
        is_stable = True
        all_states = self.policy.keys()
        for s in all_states:
            old_pi = copy.deepcopy(self.policy[s])
            argmax_action = None
            max_a_value = -1

            for action in self.actions.values():
                a_value = 0
                actionl, probs = self.get_action_probs(action)
                for a, p in zip(actionl,probs):
                    s_, r = self.env.step(s, a)
                    x_, y_ = s_
                    a_value += p*(self.gamma * self.v[y_][x_])

                if a_value > max_a_value:

                    max_a_value = a_value
                    argmax_action = action
                    self.policy[s] = argmax_action

            if old_pi != argmax_action:
                is_stable = False

        return is_stable


if __name__ == '__main__':
    env = GridWorld()
    gamma = 0.99
    is_stable = False
    count = 0

    agent = Agent(env, gamma)
    print("Iterations:")
    while is_stable == False:
          agent.evaluate_policy()
          is_stable = agent.improve_policy()
          count += 1
          print(count)

    print("Values for each state:")
    print(agent.v)
    print()
    print("Agent policy:")
    pprint.pprint((agent.policy))

    df_dict = {str(key): value for key, value in agent.value_history.items()}
    df = pd.DataFrame.from_dict(df_dict)
    df.loc[0] = 0
    df.to_csv(r"C:\Users\estee\Desktop\SC4003-Intelligent-Agents\assignment_1\policy_iteration.csv", index=False)
