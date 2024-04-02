import os

# os.environ["OMP_NUM_THREADS"] = "1"
from oniakIO import odats, ojson
import numpy as np
from lp_solver import compute_lp
import sys
import gurobipy
import cvxpy as cp
import time
from scipy.stats import levy_stable

""" Version 2 code for computing Lp subspace ground truth.
    Use ADAM lib
    Date Apr 23, 2023 """

if len(sys.argv) < 2:
    print("Please specify config json file.")
    exit(1)

config = ojson.load_config(sys.argv[1])

data = odats.ONIAKReader(config["data file"])
test = odats.read_file(config["query file"])
# if rank = 9, then we use 9-dim subspaces as data
subspRank = config["subspace rank"]
sizes = data.get_size()
if sizes == -1:
    raise  FileExistsError(config["data file"] + " does not exist.")

numSubspaces = config.get("number subspaces", sizes[0])
numSubspaces = min(numSubspaces, sizes[0] // subspRank)
d = config["dimension"]
qn = config["number queries"]
dataset = config["dataset"]
proj_d = config["projection dimension"]

data_projected = None
if sizes[1] == proj_d:
    data_projected = True
elif sizes[1] == d:
    data_projected = False
else:
    raise ValueError("Error: Data dimension neither dim or proj dim.")

seed = config.get("seed", 142857)
# id of the first query, used for coordinating multi-processing
qstart = config.get("query start id", 0)
qend = min(test.shape[0], qn + qstart)
LP = config.get("norm p", 1)
output_path = config["output file"]
# interval for each console print, in seconds
print_interv = config.get("print interval", 10)
log_file = config.get("log file", "")

if os.path.exists(output_path):
    print(output_path, "already exists, aborting..")
    exit(0)

if proj_d < subspRank:
    raise ValueError("Error: projection dim less than subspace rank. Aborted.")

assert round(LP, 4) >= 1.0
solver = "gurobi" if round(LP, 4) == round(1.0, 4) else "adam"

results = np.zeros((qn, numSubspaces))
rng = np.random.default_rng(seed)
scipy_rng = levy_stable
scipy_rng.random_state = rng
# no projection, computing ground truth
if d == proj_d:
    proj_mat = np.eye(d)
else:
    proj_mat = scipy_rng.rvs(alpha=LP, beta=0, size=(d, proj_d))

start_time = time.time()
last_time = start_time
test = test @ proj_mat
if solver == "gurobi":
    env_g = gurobipy.Env()
    env_g.setParam("OutputFlag", 0)
    env_g.setParam("Threads", 1)

for j in range(qstart, qend):
    y = test[j, :]
    if solver == "adam":
        y = y / 1000
        y = y.reshape((-1, 1))

    for i in range(numSubspaces):
        cur_time = time.time()
        if cur_time - last_time > print_interv:
            last_time = cur_time
            print(
                "lp dataset: {}, query: {}, rank:{}, lp: {}, data: {}, time: {}".format(
                    dataset, j, subspRank, LP, i, cur_time - start_time
                )
            )

        A = data.readline(subspRank)
        if not data_projected:
            A = A @ proj_mat
        if solver == "gurobi":
            A = A.T
            x = cp.Variable(subspRank)
            prob = cp.Problem(cp.Minimize(cp.norm(A @ x - y, LP)))
            results[j - qstart, i] = prob.solve(
                solver=cp.GUROBI, env=env_g, verbose=False
            )
        else:
            results[j - qstart, i] = compute_lp(A, y, LP)
    data.reset()


elapsed_time = time.time() - start_time
if log_file:
    with open(log_file, "w") as fout:
        fout.write("dataset\t#data\t#query\ttime\n")
        fout.write(
            "{}\t{}\t{}\t{}\n".format(
                dataset, numSubspaces, qend - qstart, elapsed_time
            )
        )

odats.write_file(output_path, results, dtype=odats.odat.float32)
print("Results written to ", output_path)
