from oniakIO import odats, ojson
import os, pathlib, sys
import numpy as np
from glob import glob
import pandas as pd
from collections import Counter

config = ojson.load_config(sys.argv[1])

root_file = pathlib.Path(config["root path"])
dataset = config["dataset"]
rank = config["subspace rank"]
lp = config["norm p"]
k = config["k"]
nq = config["number queries"]
l = config["number experiments"]
t = config["threshold"]
recall = config["recall"]
sol = config["solution"]
pdim = config["projection dimension"]
dim = config["dimension"]
output_path = config["output file"]

lp_code = "1" if round(lp, 4) == 1 else str(int(lp * 10))
rank_code = "" if rank == 9 else "_r{}".format(rank)


def get_file_name(query):
    return str(
        root_file
        / "gnd_data"
        / dataset
        / "{}_l{}{}_q{}.odat".format(dataset, lp_code, rank_code, query)
    )


topk_name = str(
    root_file
    / "gnd_data"
    / "{}_l{}{}_top{}.odat".format(dataset, lp_code, rank_code, k)
)


# read and store ground truth
def get_gt():
    #if os.path.exists(topk_name):
    #    return odats.read_file(topk_name)

    gnd = np.empty((nq, k), dtype=np.int32)
    gnd_dist = None
    for i in range(nq):
        dist = odats.read_file(get_file_name(i))
        if gnd_dist is None:
            n = dist.shape[1]
            gnd_dist = np.empty((nq, n))
        gnd_dist[i] = dist
        argp = np.argpartition(dist, k)[0, :k]
        # top k in argp order
        dist2 = dist[0, argp]
        arg2 = np.argsort(dist2)  # the ids are in argp order
        gnd[i] = argp[arg2]
    odats.write_file(topk_name, gnd)
    return gnd, gnd_dist


gnd, gnd_dist = get_gt()


def get_projected(query):
    return str(
        root_file
        / "result2"
        / dataset
        / "{}_l{}{}_r{}_s*_q{}_pd{}.odat".format(
            dataset, lp_code, sol, rank, query, pdim
        ),
    )


num_subspaces = 0

def get_threshold():
    dist = None
    thresholds = np.empty((nq, k))

    for q in range(nq):
        path = get_projected(q)
        if len(glob(path)) != l:
            raise FileExistsError(
                "Number of files in pattern {} is not equal to {}, actual number is {}".format(
                    path, l, len(glob(path))
                )
            )

        for i, fpath in enumerate(glob(path)[:l]):
            dist_t = odats.read_file(fpath)
            if dist is None:
                dist = np.empty((l, dist_t.shape[1]))
            dist[i] = dist_t
        gt = gnd[q]
        gt_dist = gnd_dist[q]
        assert k == len(gt)
        
        dist = np.sort(dist, axis=0)
        picked_dist = dist[t-1]
        ids_in_order = np.argsort(picked_dist)

        cur = 0
        cur_k = 0
        min_dist = [float('inf')] * k
        def update_min_dist(min_dist, x):
            if x < min_dist[-1]:
                min_dist[-1] = x
            min_dist = np.sort(min_dist)
            return min_dist

        while cur < len(ids_in_order)-1 and cur_k < k:
            id = ids_in_order[cur]
            min_dist = update_min_dist(min_dist, gt_dist[id])
            if id in gt:
                thresholds[q, cur_k] =  picked_dist[id] / min_dist[-1]
                cur_k += 1
            cur += 1

    global num_subspaces
    num_subspaces = dist.shape[1]
    return thresholds

thresholds = get_threshold()
thresholds = np.sort(thresholds.flatten())
recall_qs = [int(x * len(thresholds)) for x in recall]
thresholds = thresholds[recall_qs]
thresholds = np.stack((recall, thresholds))
print(output_path, thresholds, file=sys.stderr)
odats.write_file(output_path, thresholds)
