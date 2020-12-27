seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name ucb1 --algorithm ucb1 --sigma $1 --lambda $2
seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name gpucb --algorithm gpucb --sigma $1 --lambda $2
seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name ftl --algorithm ftl --sigma $1 --lambda $2
seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name optimist --algorithm optimist --sigma $1 --lambda $2
seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name randomist_1.1 --algorithm randomist --sigma $1 --lambda $2 --pseudo_rewards_per_timestep 1.1
seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name randomist_th --algorithm randomist --sigma $1 --lambda $2 --pseudo_rewards_per_timestep 8.1
seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name gaussian --algorithm gaussian --sigma $1 --lambda $2
seq 20 | parallel -n0 --delay 2 --ungroup --jobs 20 python run_my_experiment.py --logdir log/my_sigma$1_lambda$2 --exp_name phe_th --algorithm phe --sigma $1 --lambda $2 --pseudo_rewards_per_timestep 2.1
