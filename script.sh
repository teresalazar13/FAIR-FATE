N_RUNS=10
N_ROUNDS=100
DATASET=compas

ALPHA=0.5
for BETA0 in 0.8 0.9 0.99
do
  for L in 0.25 0.5 0.75
  do
    python3 main.py --dataset $DATASET --fl ablation_fair_demon_fixed --beta0 $BETA0 --l $L --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
  done
done

ALPHA=1.0
for BETA0 in 0.8 0.9 0.99
do
  for L in 0.25 0.5 0.75
  do
    python3 main.py --dataset $DATASET --fl ablation_fair_demon_fixed --beta0 $BETA0 --l $L --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
  done
done

for BETA0 in 0.8 0.9 0.99
do
  for L in 0.25 0.5 0.75
  do
    python3 main.py --dataset $DATASET --fl ablation_fair_demon_fixed --beta0 $BETA0 --l $L --n_runs $N_RUNS --n_rounds $N_ROUNDS
  done
done

#python3 main.py --dataset $DATASET --fl fedavg --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
#python3 main.py --dataset $DATASET --fl fedavg_gr --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
#python3 main.py --dataset $DATASET --fl fedavg_lr --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA

#for BETA in 0.8 0.9 0.99
#do
  #python3 main.py --dataset $DATASET --fl fedmom --beta $BETA --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
#done

#for BETA0 in 0.8 0.9 0.99
#do
  #python3 main.py --dataset $DATASET --fl fed_demon --beta0 $BETA0 --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
#done

#for METRICS in SP TPR EQO
#do
  #python3 main.py --dataset $DATASET --fl fed_val --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
#done

#for BETA0 in 0.8 0.9 0.99
#do
  #for RHO in 0.04 0.05
  #do
    #for L0 in 0.1 0.5
    #do
      #for MAX in 0.8 0.9 1.0
      #do
        #for METRICS in SP TPR EQO
        #do
          #python3 main.py --dataset $DATASET --fl fair_fate --beta0 $BETA0 --rho $RHO --l0 $L0 --MAX $MAX --metrics $METRICS --n_runs $N_RUNS --n_rounds $N_ROUNDS --alpha $ALPHA
        #done
      #done
    #done
  #done
#done
