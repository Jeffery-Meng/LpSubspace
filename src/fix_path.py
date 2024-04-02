import json, pathlib, os

root_path = pathlib.Path(__file__).parent.parent

for file in os.listdir(root_path / "config2"):
    if ".json" not in file:
        continue
    with open(str(root_path / "config2" / file)) as fin:
        config = json.load(fin)
    config["root path"] = str(root_path)
    with open(str(root_path / "config2" / file), "w") as fout:
        json.dump(config, fout, indent=4)

print("Fixed root paths of config files!")
