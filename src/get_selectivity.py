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
threshold_file = config["threshold file"]

# if os.path.exists(output_path):
#     print("Result already exists, skipping")
#     exit(0)

lp_code = "1" if round(lp, 4) == 1 else str(int(lp * 10))
rank_code = "" if rank == 9 else "_r{}".format(rank)

recall_bars, thresholds = odats.read_file(threshold_file)
try:
    recall_idx = np.argwhere(np.isclose(recall_bars, recall))[0][0]
    threshold = thresholds[recall_idx]
except Exception as e:
    print("Error: Recall value not found.", file=sys.stderr)
    print("Message: {}".format(e), file=sys.stderr)
    exit(1)

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

def get_recall():
    dist = None
    nns_found = 0
    candidates_checked = 0

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
        min_dist = [float('inf')] * k
        def update_min_dist(min_dist, x):
            if x < min_dist[-1]:
                min_dist[-1] = x
            min_dist = np.sort(min_dist)
            return min_dist

        while cur < len(ids_in_order)-1:
            id = ids_in_order[cur]
            if picked_dist[id] > threshold * min_dist[-1]:
                candidates_checked += cur
                break
            min_dist = update_min_dist(min_dist, gt_dist[id])
            if id in gt:
                nns_found += 1
            cur += 1

    global num_subspaces
    num_subspaces = dist.shape[1]
    return candidates_checked / nq, nns_found / (nq * k)


avg_cand_num, actual_recall = get_recall()

times = pd.read_csv(str(root_file / "result/time_summary.csv"))

gnd_row = times.loc[
    (times["dataset"] == dataset)
    & (times["solver"] != "lstsq")
    & (times["rank"] == rank)
    & (times["lp"] == lp)
    & (times["projection dimension"] == dim)
]
gnd_time = gnd_row["time"] / gnd_row["#data"] / gnd_row["#query"]
try:
    gnd_time = gnd_time.iloc[0]
except:
    print(dataset, rank, lp, dim)
    print("Error at gnd_time")
    exit(1)

sol_code = "lstsq_v2"
if sol == "sj":
    if lp == 1:
        sol_code = "gurobi"
    else:
        sol_code = "adam"

sol_row = times.loc[
    (times["dataset"] == dataset)
    & (times["solver"] == sol_code)
    & (times["rank"] == rank)
    & (times["lp"] == lp)
    & (times["projection dimension"] == pdim)
]
sol_time = sol_row["time"] / sol_row["#data"] / sol_row["#query"]
try:
    sol_time = sol_time.iloc[0]
except:
    raise ValueError(
        "Entry not found in time_summary.csv, with keys {}, {}, {}, {}, {}".format(
            dataset, sol_code, rank, lp, dim
        )
    )

gnd_all_time = num_subspaces * gnd_time
time1 = num_subspaces * sol_time * l
time2 = avg_cand_num * gnd_time
total_time = time1 + time2
speedup = gnd_all_time / total_time
speed_percent = (gnd_all_time - total_time) / gnd_all_time * 100


result = {
    "dataset": dataset,
    "algorithm": sol,
    "rank": rank,
    "projection dimension": pdim,
    "lp": lp,
    "#data": num_subspaces,
    "#query": nq,
    "L": l,
    "T": t,
    "recall_bar": recall,
    "recall": actual_recall,
    "#candidate": avg_cand_num,
    "ground truth time": gnd_all_time,
    "time1": time1,
    "time2": time2,
    "total time": total_time,
    "speedup": speedup,
    "speedup percent": speed_percent,
}
result = pd.DataFrame(result, index=[0])
result.to_csv(output_path)
