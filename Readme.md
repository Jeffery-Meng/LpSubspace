# Fast LpSubspace Search using Random Projection

## Prerequisites

1. The L1 solver (ground truth and LDL1) relies on [GUROBI](https://www.gurobi.com/). They offer free academic licenses for installation. 

2. You alse need to install our LIBONIAK package link??, since many basic functions are based on it.

3. Our datasets can be found in ??. Our codes accepts only fvecs/bvecs format. We placed all of these files in `LpSubspace/data`. You may want to check the filenames in json config files so that the program can find them.

4. Run `src/fix_path.py` to fix paths of config files to your local computer.

## How to run the codes

1. For time measurement, run `src/measure_time.sh`.
2. For accuracy, run `src/measure_accuracy.sh`. This requires about 300GB of available disk space and `time_summary.csv` as result of time measurement.
