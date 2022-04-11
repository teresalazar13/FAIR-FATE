N_RUNS=20
N_ROUNDS=50

DATASET=dutch

#python3 main.py --dataset $DATASET --fl FedAvg --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgGR --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgLR --n_runs $N_RUNS --n_rounds $N_ROUNDS

for BETA in 0.7 0.8 0.9 0.99
do
	#python3 main.py --dataset $DATASET --fl FedMom --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS
	for RHO in 0.01 0.015 0.025 #0.035 0.04 0.045 0.047 0.05
	do
	  for METRICS in SP TPR EQO
	  do
      python3 main.py --dataset $DATASET --fl FairFate --beta $BETA --rho $RHO --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
    done
  done
done
