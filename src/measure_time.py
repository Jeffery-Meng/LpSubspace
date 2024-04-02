import os

# limit numpy to use 1 core
os.environ["OMP_NUM_THREADS"] = "1"
# MKL sets adam to single-core
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from oniakIO import odats, ojson
import gurobipy
import cvxpy as cp
from lp_solver import compute_lp
import numpy as np
from scipy.stats import levy_stable
import sys, pathlib
import time
import pandas as pd

"""
    Measure running times on sampled datasets.
    Date Sep 13, 2023 """

if len(sys.argv) < 2:
    print("Please specify config json file.")
    exit(1)

config = ojson.load_config(sys.argv[1])

# use randomly generated data and query
# if rank = 9, then we use 9-dim subspaces as data
subspRank = config["subspace rank"]
numSubspaces = config["number subspaces"]
qn = config["number queries"]
dim = config["dimension"]
seed = config["seed"]
dataset = config["dataset"]
sample_num = config.get("sample num", numSubspaces)
debug_flag = config.get("debug", False)
rng = np.random.default_rng(seed)
if dataset == "gaussian":
    data = rng.normal(size=(subspRank * numSubspaces, dim))
    query = rng.normal(size=(qn, dim))
    proj_dim = dim
else:
    data = odats.read_file(config["data file"])
    query = odats.read_file(config["query file"])
    proj_dim = config.get("projection dimensions", dim)
    if proj_dim > dim:
        print("Error: Proj dim must be less than or equal to dim. Abort.")
        exit(1)
    if numSubspaces > data.shape[0] // subspRank:
        numSubspaces = data.shape[0] // subspRank

solver = config["solver"]
# id of the first query, used for coordinating multi-processing
LP = config.get("norm p", 1)
if proj_dim != dim:
    if solver in ["gurobi", "adam"]:
        proj_mat = levy_stable.rvs(alpha=LP, beta=0, size=(dim, proj_dim))
    elif solver in ["lstsq", "lstsq_v2"]:
        proj_mat = levy_stable.rvs(alpha=LP, beta=0, size=(dim, dim))
        jlt_mat = rng.standard_normal(size=(dim, proj_dim))
        proj_mat = proj_mat @ jlt_mat
    data = data @ proj_mat
    query = query @ proj_mat

if solver == "lstsq_v2":
    if debug_flag:
        data2 = data.copy()
    # orthogonalize each subspace, this should not change P2S distances, but speeds up computation
    for i in range(numSubspaces):
        A = data[subspRank * i : subspRank * (i + 1), :]
        A = A.T  # QR needs #rows > #cols
        Q, R = np.linalg.qr(A, mode="complete")
        data[subspRank * i : subspRank * (i + 1), :] = Q[:, :subspRank].T

    

output_path = config["output file"]
par_path = pathlib.Path(output_path).parent
par_path.mkdir(parents=True, exist_ok=True)

if solver not in ["gurobi", "adam", "lstsq", "lstsq_v2"]:
    print("Unsupported solver")
    exit(1)
if dim < subspRank:
    print("Invalid parameter for dim < subspace rank")
    exit(2)

if solver == "gurobi":
    env_g = gurobipy.Env()
    env_g.setParam("OutputFlag", 0)
    env_g.setParam("Threads", 1)
total_time = 0

for j in range(qn):
    y = query[j, :]
    if solver == "adam":
        y = y / 1000
        y = y.reshape((-1, 1))
    session_time = time.time()
    if sample_num < numSubspaces:
        data_samples = rng.choice(numSubspaces, config["sample num"], replace=False)
    else:
        data_samples = list(range(numSubspaces))
    start_time = time.time()
    for i in data_samples:
        A = data[subspRank * i : subspRank * (i + 1), :]
        A = A.T
        if solver == "gurobi":
            x_var = cp.Variable(subspRank)
            prob = cp.Problem(cp.Minimize(cp.norm(A @ x_var - y, LP)))
            D = prob.solve(solver=cp.GUROBI, env=env_g, verbose=False)
        elif solver == "lstsq":
            x = np.linalg.lstsq(A, y, rcond=None)
            D = np.linalg.norm(A@x[0]-y)
        elif solver == "lstsq_v2":
            D = np.linalg.norm(A@(A.T@y)-y)  # parenthesis matters
            if debug_flag:
                A = data2[subspRank * i : subspRank * (i + 1), :]
                A = A.T
                x = np.linalg.lstsq(A, y, rcond=None)
                assert np.isclose(D, np.linalg.norm(A@x[0]-y))
        elif solver == "adam":
            D = compute_lp(A.T, y, LP)
        else:
            exit(1)
    total_time += time.time() - start_time
    print(
        "dataset: {}, rank: {},  dim: {}, total time: {}".format(
            dataset, subspRank, proj_dim, total_time
        )
    )

result = {
    "dataset": dataset,
    "solver": solver,
    "rank": subspRank,
    "projection dimension": proj_dim,
    "lp": LP,
    "#data": sample_num,
    "#query": qn,
    "time": total_time,
}
result = pd.DataFrame(result, index=[0])
result.to_csv(output_path)
