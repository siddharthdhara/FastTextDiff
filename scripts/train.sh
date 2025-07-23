

DATASET=$1 # Available datasets: qata_cov19_v2_2, monuseg_2, MosMedDataPlus

python train3.py --exp experiments/${DATASET}/ddpm.json

