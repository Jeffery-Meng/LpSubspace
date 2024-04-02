import json, subprocess, pathlib, psutil, os

repetitions = {
    1.0: {
        # rank, algorithm, pd, L
        "sift": [4, 9, 16, 32, 64],
        "deep": [9],
        "trevi": [4, 9, 16, 32, 64, 128],
        "gist": [9],
        "mnist": [9],
        "enron": [9],
    },
    1.2: {
        "sift": [9],
        "trevi": [9],
    },
}

num_threads = int(psutil.cpu_count() * 0.9)
jlt_dimension = 256

root_file = pathlib.Path(__file__).parent.parent
config_path = root_file / "config2"
config_auto_path = config_path / "auto"
gnd_path = root_file / "gnd_data"

count = 0

for lp, lp_dict in repetitions.items():
    lp_code = "1" if round(lp, 4) == 1 else str(int(lp * 10))
    for dataset, exp_list in lp_dict.items():
        config_old = config_path / "{}_l1_time.json".format(dataset)
        with open(str(config_old)) as fin:
            conf = json.load(fin)
        gnd_file = str(gnd_path / dataset)
        del conf["sample num"]
        del conf["solver"]
        del conf["projection dimensions"]
        del conf["seed"]
        del conf["number subspaces"]

        for rank in exp_list:
            rank_code = "" if rank == 9 else "_r{}".format(rank)
            ok_file = os.path.join(
                gnd_file,
                "{}_l{}{}_ok.odat".format(dataset, lp_code, rank_code),
            )
            if os.path.exists(ok_file):
                continue

            conf["max process"] = num_threads
            conf["script"][1] = "ROOT/src/compute_lp_p2s.py"
            conf["subspace rank"] = rank
            conf["number queries"] = 1
            conf["query start id"] = ["ITER_RANGE", 100]
            conf["projection dimension"] = conf["dimension"]
            conf["norm p"] = lp
            
            conf["output file"] = os.path.join(
                gnd_file,
                "{}_l{}{}_q{{query start id}}.odat".format(dataset, lp_code, rank_code),
            )

            json_file = str(config_auto_path / "gnd_GEN_{}.json".format(count))
            with open(json_file, "w") as fout:
                json.dump(conf, fout, indent=4)

            proc = subprocess.run(
                ["python3", "{}/oniakRun/auto_run_experiment.py".format(os.environ["ONIAKLIB"]), json_file]
            )
            if proc.returncode == 0:
                with open(ok_file, "w") as fout:
                    pass
            count += 1
