import json, subprocess, pathlib, psutil, os

RECALL = 0.89
recall_code = int(RECALL * 100)

repetitions = {
    1.0: {
        # rank, algorithm, pd, L, T
        "sift": [
            (4, "sj", 25, 1, 1),
            (4, "jl", 128, 12, 3),
            (9, "sj", 25, 1, 1),
            (9, "jl", 128, 12, 3),
            (16, "sj", 25, 1, 1),
            (16, "jl", 128, 10, 2),
            (32, "sj", 48, 1, 1),
            (32, "jl", 128, 14, 3),
            (64, "sj", 80, 1, 1),
            (64, "jl", 128, 11, 4),
        ],
        "deep": [(9, "sj", 25, 1, 1), (9, "jl", 96, 12, 3)],
        "trevi": [
            (4, "sj", 64, 4, 1),
            (4, "jl", 4096, 86, 10),
            (4, "jlt", 256, 83, 16),
            (9, "sj", 32, 12, 1),
            (9, "jl", 4096, 93, 10),
            (9, "jlt", 256, 91, 15),
            (16, "sj", 64, 9, 1),
            (16, "jl", 4096, 81, 10),
            (16, "jlt", 256, 83, 10),
            (32, "sj", 64, 7, 1),
            (32, "jl", 4096, 87, 10),
            (32, "jlt", 256, 83, 15),
            (64, "sj", 96, 17, 1),
            (64, "jl", 4096, 75, 12),
            (64, "jlt", 256, 90, 10),
            (128, "sj", 192, 12, 1),
            (128, "jl", 4096, 89, 10),
            (128, "jlt", 256, 86, 23),
        ],
        "gist": [(9, "sj", 64, 1, 1), (9, "jl", 960, 45, 7), (9, "jlt", 128, 56, 10)],
        "mnist": [(9, "sj", 64, 1, 1), (9, "jl", 784, 24, 2), (9, "jlt", 128, 25, 2)],
        "enron": [(9, "sj", 64, 1, 1), (9, "jl", 1369, 91, 8), (9, "jlt", 128, 77, 8)],
    },
    1.2: {
        "sift": [(9, "sj", 25, 1, 1), (9, "jl", 128, 10, 3)],
        "trevi": [(9, "sj", 64, 2, 1), (9, "jl", 4096, 42, 7), (9, "jlt", 256, 42, 8)],
    },
}

num_threads = int(psutil.cpu_count() * 0.9)
jlt_dimension = 256

root_file = pathlib.Path(__file__).parent.parent
config_path = root_file / "config2"
config_auto_path = config_path / "auto"

# this part of code is only intended for the author, because his SSD does not have enough space
if str(root_file) == "/media/gtnetuser/SSD_2TB_BEST/LpSubspace":
    external_file = pathlib.Path("/media/mydrive/LpSubspace")
else:
    external_file = root_file
internal_file = root_file / "result3"
thresh_file = root_file / "result2"
count = 0

for lp, lp_dict in repetitions.items():
    lp_code = "1" if round(lp, 4) == 1 else str(int(lp * 10))
    for dataset, exp_list in lp_dict.items():
        config_old = config_path / "{}_l1_time.json".format(dataset)
        with open(str(config_old)) as fin:
            conf = json.load(fin)
        internal_file2 = str(internal_file / dataset)
        thresh_file2 = str(thresh_file / dataset)
        del conf["sample num"]
        del conf["solver"]
        del conf["projection dimensions"]

        for rank, solution, proj_dim, bigl, threshold in exp_list:
            conf["max process"] = num_threads
            data_code = "jl" if solution == "jlt" else ""
            conf["script"][1] = "ROOT/src/get_selectivity.py"
            conf["subspace rank"] = rank
            conf["projection dimension"] = proj_dim
            conf["norm p"] = lp
            conf["k"] = 10
            conf["solution"] = solution
            conf["recall"] = RECALL
            conf["number experiments"] = bigl
            conf["threshold"] = threshold
            conf["threshold file"] = os.path.join(
                thresh_file2,
                "{}_l{}{}_r{}_pd{}_threshold.odat".format(
                    dataset, lp_code, solution, rank, proj_dim
                ),
            )
            conf["output file"] = os.path.join(
                internal_file2,
                "{}_l{}{}_r{}_pd{}_rcl{}.csv".format(
                    dataset, lp_code, solution, rank, proj_dim, recall_code
                ),
            )

            json_file = str(
                config_auto_path / "selectivity_GEN_{}.json".format(count)
            )
            with open(json_file, "w") as fout:
                json.dump(conf, fout, indent=4)

            subprocess.run(
                ["python3", "{}/oniakRun/auto_run_experiment.py".format(os.environ["ONIAKLIB"]), json_file]
            )
            count += 1
