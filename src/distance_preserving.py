""" Run distance preserving experiment on SIFT L1 rank 9"""

from oniakIO import odats
from sys import argv
import numpy as np
from scipy.stats import levy_stable
import pathlib

NUM_QUERY = 100
RANK = 9
LP = 1
DIM = 128
NUM_DATA = 100
NUM_SAMPLE = 100    # 100 sample data items per query
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

def calc_l2(A, y):
    A = A.T
    x = np.linalg.lstsq(A,y,rcond=None)
    return np.linalg.norm(A@x[0]-y)

result = np.empty((NUM_DATA,NUM_QUERY), dtype=np.float32)
for cid in range(NUM_DATA):
  for qid in range(NUM_QUERY):
     result[cid, qid] = calc_l2(data[cid*RANK:(cid+1)*RANK], query[qid])

odats.write_file(RESULT_PATH, result)

