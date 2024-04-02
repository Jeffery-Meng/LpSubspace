# at project root 
BASEDIR=$(dirname "$0")
cd $BASEDIR/..

python3 ${ONIAKLIB}/oniakRun/auto_run_experiment.py config2/deep_l1_time.json \
config2/enron_l1_time.json config2/gist_l1_time.json config2/sift_l1_time.json config2/trevi_l1_time.json \
config2/mnist_l1_time.json config2/sift_l12_time.json config2/trevi_l12_time.json

python3 src/summarize.py 'result/*time/*.csv' result/time_summary.csv