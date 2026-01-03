
import random
import numpy as np
from hierarchy import Hierarchy

def generate_hierarchy(depth, branching_factor):
    np.random.seed(42)
    random.seed(42)
    structure = {}
    current_layer = ['Total']
    node_ctr = 1
    for d in range(depth):
        next_layer = []
        for parent in current_layer:
            # Force n_children to be close to branching_factor for testing
            n_children = random.randint(branching_factor-1, branching_factor)
            children = []
            for _ in range(n_children):
                child_name = f'Node_{d+1}_{node_ctr}'
                children.append(child_name)
                node_ctr += 1
            structure[parent] = children
            next_layer.extend(children)
        current_layer = next_layer
    return Hierarchy(structure)

print("Testing Hierarchy Sizes...")
configs = [
    (4, 5),
    (5, 3),
    (5, 4),
    (6, 3),
    (7, 2),
]

for d, b in configs:
    h = generate_hierarchy(d, b)
    print(f"D={d}, B={b} -> Nodes: {h.n_nodes}")
