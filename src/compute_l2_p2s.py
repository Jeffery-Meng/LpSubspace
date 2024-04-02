import os
from oniakIO import odats, ojson
import numpy as np
import sys, json
import time
from scipy.stats import levy_stable

os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

""" Computing Lp subspace ground truth.
    If jlt is not used, set jlt dimension to data dimension
   """

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
jlt_dim = config["projection dimension"]
assert(jlt_dim == sizes[1])
data_projected = config.get("data projected", True)
qn = config["number queries"]
dataset = config["dataset"]
seed = config["seed"]

# id of the first query, used for coordinating multi-processing
qstart = config.get("query start id", 0)
qend = min(test.shape[0], qn + qstart)
LP = config.get("norm p", 1)
output_path = config["output file"]
# interval for each console print, in seconds
print_interv = config.get("print interval", 10)
log_file = config.get("log file", "")

if os.path.exists(output_path):
    print(output_path, "already computed. Skipping.")
    exit(0)

if jlt_dim < subspRank:
    raise ValueError("Error: projection dim less than subspace rank. Aborted.")

results = np.zeros((qn,numSubspaces))
rng = np.random.default_rng(seed)
scipy_rng = levy_stable
scipy_rng.random_state = rng
proj_mat = scipy_rng.rvs(alpha=LP, beta=0, size=(d, d))
test = test @ proj_mat
# the JL random matrix has been directly generated from seed at preprocessing step
rng = np.random.default_rng(seed)
if jlt_dim != d:
    jlt_mat = rng.standard_normal(size=(d,jlt_dim))
    test = test @ jlt_mat

start_time = time.time()
last_time = start_time

for j in range(qstart, qend):
    y = test[j,:]
    
    session_time = time.time()
    for i in range(numSubspaces):
        cur_time = time.time()
        if cur_time - last_time > print_interv:
            last_time = cur_time
            print(
                "l2 dataset: {}, query: {}, rank:{}, lp: {}, data: {}, time: {}".format(
                    dataset, j, subspRank, LP, i, cur_time - start_time
                )
            )
        A = data.readline(subspRank) 
        if not data_projected:
            A = A @ proj_mat
        A = A.T
        x = np.linalg.lstsq(A,y,rcond=None)
        results[j-qstart,i] = np.linalg.norm(A@x[0]-y)
    
    cur_time = time.time()
    data.reset()
    print("dataset: {}, query: {}, time: {}".format(dataset, j, cur_time - start_time))
        
odats.write_file(output_path, results, dtype=odats.odat.float32)