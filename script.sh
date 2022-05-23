N_RUNS=10
N_ROUNDS=100

for DATASET in compas law
do
  for BETA in 0.8 0.9 0.99
  do
    python3 main.py --dataset $DATASET --fl FedDemon --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha 0.5
    python3 main.py --dataset $DATASET --fl FedDemon --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha 1.0
    python3 main.py --dataset $DATASET --fl FedDemon --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS
  done
done

DATASET=compas
ALPHA=1.0

python3 main.py --dataset $DATASET --fl FedAvg --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
python3 main.py --dataset $DATASET --fl FedAvgGR --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
python3 main.py --dataset $DATASET --fl FedAvgLR --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
python3 main.py --dataset $DATASET --fl FedMom --beta 0.8 --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA

