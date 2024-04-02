BASEDIR=$(dirname "$0")
cd $BASEDIR

python3 run_data_transform.py 2> ../summary/data_transform_log.txt
python3 run_ground_truth.py 2> ../summary/ground_truth_log.txt
python3 run_p2s_computation.py 2> ../summary/p2s_log.txt
python3 run_get_selectivity.py 2> ../summary/selectivity_log.txt
python3 summarize.py '../result3/*/*_l*_r*_pd*.csv' ../result3/accuracy_summary.csv
