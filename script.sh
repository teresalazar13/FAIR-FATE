DATASET=adult

for BETA in 0.7 0.8 0.9 0.99
do
	for LAMBDA_ in 0.035 0.04 0.045 0.047 0.05
	do
    python3 main.py --dataset $DATASET --fl FairFate --beta $BETA --lambda_ $LAMBDA_ --metrics TPR FPR --n_runs 10 --n_rounds 50
  done
done