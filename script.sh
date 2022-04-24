N_RUNS=20
N_ROUNDS=50

DATASET=compas

python3 main.py --dataset $DATASET --fl FedAvg --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
python3 main.py --dataset $DATASET --fl FedAvgGR --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
python3 main.py --dataset $DATASET --fl FedAvgLR --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS

for METRICS in SP TPR EQO
do
  python3 main.py --dataset $DATASET --fl FedVal --alpha $ALPHA --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
done

for BETA in 0.7 0.8 0.9 0.99
do
	python3 main.py --dataset $DATASET --fl FedMom --alpha $ALPHA --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS
	for RHO in 0.035 0.04 0.045 0.047 0.05
	do
	  for METRICS in SP TPR EQO
	  do
	    for L0 in 10000
	    do
        python3 main.py --dataset $DATASET --fl FairFate --alpha $ALPHA --beta $BETA --rho $RHO --l0 $L0 --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
      done
    done
  done
done


# RESULTS OVER ROUNDS
#N_RUNS=20
#N_ROUNDS=100

#DATASET=compas
#METRICS=TPR

#python3 main.py --dataset $DATASET --fl FedAvg --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgGR --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgLR --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedMom --beta 0.99 --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FairFed --beta 1 --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedVal --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FairFate --beta 0.8 --rho 0.05 --l0 0.1 --MAX 1 --metrics TPR --n_runs $N_RUNS --n_rounds $N_ROUNDS

#ALPHA=0.5
#python3 main.py --dataset $DATASET --fl FedAvg --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgGR --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgLR --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedMom --alpha $ALPHA --beta 0.9 --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FairFed --alpha $ALPHA --beta 1 --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedVal --alpha $ALPHA --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FairFate --alpha $ALPHA --beta 0.7 --rho 0.045 --l0 0.1 --MAX 1 --metrics TPR --n_runs $N_RUNS --n_rounds $N_ROUNDS

#ALPHA=0.25
#python3 main.py --dataset $DATASET --fl FedAvg --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgGR --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedAvgLR --alpha $ALPHA --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedMom --alpha $ALPHA --beta 0.8 --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FairFed --alpha $ALPHA --beta 1 --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FedVal --alpha $ALPHA --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS
#python3 main.py --dataset $DATASET --fl FairFate --alpha $ALPHA --beta 0.8 --rho 0.047 --l0 0.1 --MAX 1 --metrics TPR --n_runs $N_RUNS --n_rounds $N_ROUNDS
