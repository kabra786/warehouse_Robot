from utils import load_config, set_seed
from problem import WarehouseProblem
from state import RobotState
from search import UniformCostSearch

# Load config and set seed
config = load_config()
set_seed(config["seed"])

# Prepare grid
grid_template = [line.split() for line in config["grid_template"].strip().splitlines()]

# Find packages and delivery
packages = [(i,j) for i,row in enumerate(grid_template)
            for j,val in enumerate(row) if val=='P']
delivery = [(i,j) for i,row in enumerate(grid_template)
            for j,val in enumerate(row) if val=='D'][0]

# Initialize problem
problem = WarehouseProblem(grid_template, packages, delivery)
start_pos, collected = problem.get_initial_state()
initial_state = RobotState(start_pos, collected)

# Solve using UCS
ucs = UniformCostSearch(problem)
path, cost, nodes_expanded, elapsed = ucs.solve(initial_state)

print("Path:", path)
print("Total cost:", cost)
print("Nodes expanded:", nodes_expanded)
print("Elapsed time:", elapsed)

