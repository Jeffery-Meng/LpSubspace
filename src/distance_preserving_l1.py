""" Run distance preserving experiment on SIFT L1 rank 9"""

from oniakIO import odats
from sys import argv
import numpy as np
from scipy.stats import levy_stable
import pathlib
import gurobipy
import cvxpy as cp

NUM_QUERY = 100
RANK = 9
LP = 1
DIM = 128
NUM_DATA = 100  # 10 sample data items per query
SEED = int(argv[1])
RESULT_PATH = argv[2]

dataset = "sift"
root_path = pathlib.Path(__file__).parent.parent
data = odats.read_file(str(root_path / "data/sift100sample.odat"))
query = odats.read_file(str(root_path / "data/sift_query.odat"))[:NUM_QUERY]

rng = np.random.default_rng(SEED)
scipy_rng = levy_stable
scipy_rng.random_state = rng

proj_mat = scipy_rng.rvs(alpha=LP, beta=0, size=(DIM, DIM))
data = data @ proj_mat
query = query @ proj_mat

env_g = gurobipy.Env()
env_g.setParam("OutputFlag", 0)
env_g.setParam("Threads", 1)


def calc_l1(A, y):
    A = A.T
    x = cp.Variable(RANK)
    prob = cp.Problem(cp.Minimize(cp.norm(A @ x - y, LP)))
    return prob.solve(solver=cp.GUROBI, env=env_g, verbose=False)


result = np.empty((NUM_DATA, NUM_QUERY), dtype=np.float32)
for cid in range(NUM_DATA):
    for qid in range(NUM_QUERY):
        result[cid, qid] = calc_l1(data[cid * RANK : (cid + 1) * RANK], query[qid])

odats.write_file(RESULT_PATH, result)
