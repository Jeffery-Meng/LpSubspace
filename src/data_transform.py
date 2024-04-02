import os
from oniakIO import odats, ojson
import numpy as np
import sys, json
import time
from scipy.stats import levy_stable

os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

""" Version 2 code for computing Lp subspace ground truth.
    Use GUROBI lib for running p=1 only.
    Date Apr 23, 2023 """

if len(sys.argv) < 2:
    print("Please specify config json file.")
    exit(1)

config = ojson.load_config(sys.argv[1])

data = odats.ONIAKReader(config["data file"])
try:
    transformed_dump = odats.ONIAKWriter(config["transformed file"])
except ValueError:
    print("Already exists, skipping...")
    exit(0)
# if rank = 9, then we use 9-dim subspaces as data
subspRank = config["subspace rank"]
sizes = data.get_size()
numSubspaces = min(config["number subspaces"], sizes[0] // subspRank) # type: ignore
d = config["dimension"]
assert(d == sizes[1])
qn = config["number queries"]
dataset = config["dataset"]
seed = config["seed"]
proj_dim = config["projection dimension"]
jlt_dim = config.get("jlt dimension", -1)
jlt_valid = True
try:
    jlt_dump = odats.ONIAKWriter(config["jlt file"])
except (KeyError, ValueError):
    print("JLT file not written, skipping...")
    jlt_valid  = False

# id of the first query, used for coordinating multi-processing
qstart = config.get("query start id", 0)
LP = config.get("norm p", 1)
# interval for each console print, in seconds
print_interv = config.get("print interval", 10)
log_file = config.get("log file", "")

results = np.zeros((qn,numSubspaces))
rng = np.random.default_rng(seed)
scipy_rng = levy_stable
scipy_rng.random_state = rng

if jlt_dim == -1:
    proj_mat = scipy_rng.rvs(alpha=LP, beta=0, size=(d, proj_dim))
    jlt_valid = False
else:
    proj_mat = scipy_rng.rvs(alpha=LP, beta=0, size=(d, d))
    rng = np.random.default_rng(seed)
    jlt_mat = rng.standard_normal(size=(d,jlt_dim))

start_time = time.time()
last_time = start_time


for i in range(numSubspaces+1):
    cur_time = time.time()
    if cur_time - last_time > print_interv:
        last_time = cur_time
        print("dataset: {}, data: {}, time: {}".format(dataset, i, cur_time - start_time))

    A = data.readline(subspRank) @ proj_mat
    transformed_dump.writelines(A)
    if jlt_valid:
        A = A @ jlt_mat
        jlt_dump.writelines(A)

cur_time = time.time()
data.reset()
print("dataset: {}, time: {}".format(dataset, cur_time - start_time))