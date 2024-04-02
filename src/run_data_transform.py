import json, subprocess, pathlib, psutil, os, glob
from oniakIO import odats
from oniakPath import valid_count, opath

repetitions = {
    1.0: {
        "sift": {25: 1, 48: 1, 80: 1, 128: 14},
        "deep": {25: 1, 96: 12},
        "trevi": {32: 12, 64: 9, 96: 17, 192: 12, 4096: 93},
        "gist": {64: 1, 960: 56},
        "mnist": {64: 1, 784: 25},
        "enron": {64: 1, 1369: 91},
    },
    1.2: {"sift": {25: 1, 128: 10}, "trevi": {64: 2, 4096: 42}},
}

num_threads = int(psutil.cpu_count() * 0.9)

root_file = pathlib.Path(__file__).parent.parent
root_file = root_file.resolve()
config_path = root_file / "config2"
config_auto_path = config_path / "auto"

# this part of code is only intended for the author, because his SSD does not have enough space
if str(root_file) == "/media/gtnetuser/SSD_2TB_BEST/LpSubspace":
    external_file = pathlib.Path("/media/mydrive/LpSubspace")
else:
    external_file = root_file
external_file = external_file / "result3"

for lp, lp_dict in repetitions.items():
    lp_code = str(int(lp * 10))
    for dataset, proj_dict in lp_dict.items():
        config_old = config_path / "{}_l1_time.json".format(dataset)
        with open(str(config_old)) as fin:
            conf = json.load(fin)
        external_file2 = str(external_file / dataset)
        del conf["sample num"]
        del conf["solver"]
        del conf["output file"]
        del conf["projection dimensions"]

        data_path = opath.change_to_absolute(conf["data file"], conf)
        raw_data = odats.ONIAKReader(data_path)
        num_data, dim = raw_data.get_size()

        for proj_dim, bigl in proj_dict.items():
            conf["max process"] = num_threads
            conf["script"][1] = "ROOT/src/data_transform.py"
            conf["subspace rank"] = 4096

            existing_files = os.path.join(
                external_file2, "{}_l{}_s*_pd{}.odat".format(dataset, lp_code, proj_dim)
            )
            num_existing = valid_count.check_files(
                existing_files, (num_data, proj_dim), delete=True
            )

            jlt_dimension = 256 if dataset == "trevi" else 128
            if proj_dim > jlt_dimension:
                existing_jlt = os.path.join(
                    external_file2,
                    "{}_l{}jl_s*_pd{}.odat".format(dataset, lp_code, jlt_dimension),
                )
                jlt_existing = valid_count.check_files(
                    existing_jlt, (num_data, jlt_dimension), delete=True
                )
                num_existing = min(num_existing, jlt_existing)
            num_left = bigl - num_existing

            if num_left <= 0:
                # all jobs are finished, skipping
                print(dataset, "completed dim", proj_dim, "L", bigl)
                continue
            print(external_file2,  num_existing)
            print(dataset, proj_dim, "remaining", num_left)

            conf["seed"] = ["ITER_RANDOM", num_left]
            conf["number queries"] = 1

            conf["projection dimension"] = proj_dim
            conf["norm p"] = lp
            conf["transformed file"] = os.path.join(
                external_file2,
                "{}_l{}_s{{seed}}_pd{}.odat".format(dataset, lp_code, proj_dim),
            )

            if proj_dim > jlt_dimension:
                conf["jlt dimension"] = jlt_dimension
                conf["jlt file"] = os.path.join(
                    external_file2,
                    "{}_l{}jl_s{{seed}}_pd{}.odat".format(
                        dataset, lp_code, jlt_dimension
                    ),
                )

            json_file = str(
                config_auto_path / "{}_l{}_pd{}.json".format(dataset, lp_code, proj_dim)
            )
            with open(json_file, "w") as fout:
                json.dump(conf, fout, indent=4)

            subprocess.run(
                [
                    "python3",
                    "{}/oniakRun/auto_run_experiment.py".format(os.environ["ONIAKLIB"]),
                    json_file,
                ]
            )
