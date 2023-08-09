#!/bin/bash

# ---- Problem Definition ---- #
# Number of parameters
n_param=30
# Parameter limits
lower_limit_1=$(echo "40.0")
upper_limit_1=$(echo "1960.0")

# ---- Optimiser - General ---- #
# Number of Gen to save state / Communications
checkpoint=1

# ---- Repository ---- #
# [0,1] max porcentage of population from rep
max_pop_from_rep=$(echo "1.0")
# [0,1] Probability of initialise from rep
init_prob=$(echo "0.5")
# Repository size
n_rep=100

# ---- Stall mechanism  ---- #
# Activate wrapper kill mechanism
kill_flag=True
# Parameter to define number of generations allowed
p_n=3
# Max number of checkpoints reached without improving fmin
n_0=2
# Number of best runs allowed to run max generations
n_best_runs=2
# Stall tolerance level (0, 1) - closer to one more exploration
stall_tol=$(echo "0.001")
# Number of stall error for each optim - higher number more exploration
n_stall=4
# Max number of runs
max_runs=10000

# Experiment ID
exp_id=$2

# ---- MPI ---- #
# Number of devices: n_workers + 1 (supervisor)
n_devices=5

# ---- Report ---- #
# Report flag (0: optimise, 1: report)
report=$1

if [ "$report" -eq 0 ]; then
  # ---- Run to Optimise ---- #
  mpirun -n $n_devices python3 main.py --n_devices $n_devices --report $report \
  --n_param $n_param --lower_limit_1 $lower_limit_1 --upper_limit_1 $upper_limit_1 \
  --checkpoint $checkpoint --max_pop_from_rep $max_pop_from_rep --n_rep $n_rep \
  --init_prob $init_prob --kill_flag $kill_flag --p_n $p_n --n_0 $n_0 \
  --n_best_runs $n_best_runs --stall_tol $stall_tol --n_stall $n_stall  \
  --max_runs $max_runs \
  --exp_id $exp_id
else
  # ---- Run to Report ---- #
  python3 main.py --report $report \
  --n_param $n_param --lower_limit_1 $lower_limit_1 --upper_limit_1 $upper_limit_1 \
  --checkpoint $checkpoint --max_pop_from_rep $max_pop_from_rep --n_rep $n_rep \
  --init_prob $init_prob --kill_flag $kill_flag --p_n $p_n --n_0 $n_0 \
  --n_best_runs $n_best_runs --stall_tol $stall_tol --n_stall $n_stall  \
  --exp_id $exp_id --work_out True
fi

