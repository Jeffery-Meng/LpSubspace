import json, subprocess, pathlib, psutil, os

repetitions = {
    1.0: {
        # rank, algorithm, pd, L
        "sift": [
            (4, "sj", 25, 1),
            (4, "jl", 128, 12),
            (9, "sj", 25, 1),
            (9, "jl", 128, 12),
            (16, "sj", 25, 1),
            (16, "jl", 128, 10),
            (32, "sj", 48, 1),
            (32, "jl", 128, 14),
            (64, "sj", 80, 1),
            (64, "jl", 128, 11),
        ],
        "deep": [(9, "sj", 25, 1), (9, "jl", 96, 12)],
        "trevi": [
            (4, "sj", 64, 4),
            (4, "jl", 4096, 86),
            (4, "jlt", 256, 83),
            (9, "sj", 32, 12),
            (9, "jl", 4096, 93),
            (9, "jlt", 256, 91),
            (16, "sj", 64, 9),
            (16, "jl", 4096, 81),
            (16, "jlt", 256, 83),
            (32, "sj", 64, 7),
            (32, "jl", 4096, 87),
            (32, "jlt", 256, 83),
            (64, "sj", 96, 17),
            (64, "jl", 4096, 75),
            (64, "jlt", 256, 90),
            (128, "sj", 192, 12),
            (128, "jl", 4096, 89),
            (128, "jlt", 256, 86),
        ],
        "gist": [(9, "sj", 64, 1), (9, "jl", 960, 45), (9, "jlt", 128, 56)],
        "mnist": [(9, "sj", 64, 1), (9, "jl", 784, 24), (9, "jlt", 128, 25)],
        "enron": [(9, "sj", 64, 1), (9, "jl", 1369, 91), (9, "jlt", 128, 77)],
    },
    1.2: {
        "sift": [(9, "sj", 25, 1), (9, "jl", 128, 10)],
        "trevi": [(9, "sj", 64, 2), (9, "jl", 4096, 42), (9, "jlt", 256, 42)],
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
external_file = external_file / "result3"
internal_file = root_file / "result3"
count = 0

for lp, lp_dict in repetitions.items():
    lp_code_old = str(int(lp * 10))
    lp_code = "1" if round(lp, 4) == 1 else lp_code_old
    for dataset, exp_list in lp_dict.items():
        config_old = config_path / "{}_l1_time.json".format(dataset)
        with open(str(config_old)) as fin:
            conf = json.load(fin)
        external_file2 = str(external_file / dataset)
        internal_file2 = str(internal_file / dataset)
        del conf["sample num"]
        del conf["solver"]
        del conf["projection dimensions"]

        for rank, solution, proj_dim, bigl in exp_list:
            ok_file = os.path.join(
                internal_file2,
                "{}_l{}{}_r{}_pd{}_ok.odat".format(
                    dataset, lp_code, solution, rank, proj_dim
                ),
            )
            if os.path.exists(ok_file):
                continue

            conf["max process"] = num_threads
            solver_script = "lp" if solution == "sj" else "l2"
            data_code = "jl" if solution == "jlt" else ""
            conf["script"][1] = "ROOT/src/compute_{}_p2s.py".format(solver_script)
            conf["subspace rank"] = rank
            conf["data file"] = [
                "ITER_FILE",
                os.path.join(
                    external_file2,
                    "{}_l{}{}_s*_pd{}.odat".format(dataset, lp_code_old, data_code, proj_dim),
                ),
                bigl,
            ]
            conf["seed"] = ["ITER_EXTRACT", "data file", "s"]
            conf["number queries"] = 1
            # just a large enough number to use all input data
            conf["number subspaces"] = 999999
            conf["query start id"] = ["ITER_RANGE", 100]
            conf["projection dimension"] = proj_dim
            conf["norm p"] = lp
            conf["output file"] = os.path.join(
                internal_file2,
                "{}_l{}{}_r{}_s{{seed}}_q{{query start id}}_pd{}.odat".format(
                    dataset, lp_code, solution, rank, proj_dim
                ),
            )

            json_file = str(
                config_auto_path / "computation_GEN_{}.json".format(count)
            )
            with open(json_file, "w") as fout:
                json.dump(conf, fout, indent=4)

            proc = subprocess.run(
                ["python3", "{}/oniakRun/auto_run_experiment.py".format(os.environ["ONIAKLIB"]), json_file]
            )
            if proc.returncode == 0:
                with open(ok_file, "w") as fout:
                    pass
            else:
                if os.path.exists(ok_file):
                    os.remove(ok_file)
            count += 1
