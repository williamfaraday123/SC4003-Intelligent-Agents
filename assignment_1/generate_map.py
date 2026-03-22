import numpy as np
import copy
import pprint

def generate_map(n):
    # Define the colors and wall as strings
    elements = ["Brown", "White", "Wall", "Green"]

    # Generate a NxN 2D array with random selection of the elements
    map_nxn = np.random.choice(elements, size=(n, n), p=[0.25, 0.5, 0.15, 0.1])

    map_list = map_nxn.tolist()
    return map_list


bonus = generate_map(10)
for i in bonus:
    print(i)