N_RUNS=10
N_ROUNDS=100
DATASET=compas
ALPHA=1.0

for BETA in 0.85 0.95 #0.8 0.9 0.99
do
  for RHO in 0.02 0.03 0.06 #0.04 0.05
  do
    for L0 in 0.2 0.3 0.4 #0.1 0.5
    do
      for MAX in 0.85 0.95 #0.8 0.9 1.0
      do
        for METRICS in SP TPR EQO
        do
          python3 main.py --dataset $DATASET --fl FairFate --beta $BETA --rho $RHO --l0 $L0 --MAX $MAX --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
        done
      done
    done
  done
done
