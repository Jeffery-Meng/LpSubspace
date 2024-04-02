BASEDIR=$(dirname "$0")
cd $BASEDIR

python3 run_data_transform.py 2> ../summary/data_transform_log.txt
python3 run_ground_truth.py 2> ../summary/ground_truth_log.txt
python3 run_p2s_computation.py 2> ../summary/p2s_log.txt
python3 run_get_threshold.py 2> ../summary/threshold_log.txt

# delete intermediate files except computed thresholds
python3 $ONIAKLIB/oniakPath/odelete.py ../result2 '^(?!.*threshold\.odat)'
