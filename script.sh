DATASET=adult
ALPHA=0.25

python3 main.py --dataset $DATASET --fl FedAvg --alpha $ALPHA --n_runs 10 --n_rounds 50
python3 main.py --dataset $DATASET --fl FedAvgGR --alpha $ALPHA --n_runs 10 --n_rounds 50
python3 main.py --dataset $DATASET --fl FedAvgLR --alpha $ALPHA --n_runs 10 --n_rounds 50

for BETA in 0.7 0.8 0.9 0.99
do
	python3 main.py --dataset $DATASET --fl FedMom --alpha 0.25 --beta $BETA --n_runs 10 --n_rounds 50
	for LAMBDA_ in 0.035 0.04 0.045 0.047 0.05
	do
	  for METRICS in SP TRP EQO
	  do
      python3 main.py --dataset $DATASET --fl FairFate --alpha 0.25 --beta $BETA --lambda_ $LAMBDA_ --metrics $METRICS --n_runs 10 --n_rounds 50
    done
  done
done
