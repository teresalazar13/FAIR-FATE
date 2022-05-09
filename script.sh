N_RUNS=10
N_ROUNDS=100
DATASET=compas
DATASET=law
ALPHA=1.0

python3 main.py --dataset $DATASET --fl FedAvg --n_runs $N_RUNS --n_rounds $N_ROUNDS #--alpha $ALPHA
python3 main.py --dataset $DATASET --fl FedAvgGR --n_runs $N_RUNS --n_rounds $N_ROUNDS #--alpha $ALPHA
python3 main.py --dataset $DATASET --fl FedAvgLR --n_runs $N_RUNS --n_rounds $N_ROUNDS #--alpha $ALPHA

for BETA in 0.8 0.9 0.99
do
  python3 main.py --dataset $DATASET --fl FedMom --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS #--alpha $ALPHA
done

for BETA in 0.8 0.9 0.99
do
  python3 main.py --dataset $DATASET --fl FedDemon --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS #--alpha $ALPHA
done

for METRICS in SP TPR EQO
do
  python3 main.py --dataset $DATASET --fl FedVal --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS #--alpha $ALPHA
done

for BETA in 0.8 0.9 0.99
do
  for RHO in 0.04 0.05
  do
    for L0 in 0.1 0.5
    do
      for MAX in 0.8 0.9 1.0
      do
        for METRICS in SP TPR EQO
        do
          python3 main.py --dataset $DATASET --fl FairFate --beta $BETA --rho $RHO --l0 $L0 --MAX $MAX --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS #--alpha $ALPHA
        done
      done
    done
  done
done
