#!/bin/bash
count=0
python3 ./MedMNIST/VNNLIBmakerOCT.py
while [ $count -le 295 ]; do
    python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST.yaml --model OCTMNIST_FC --cex_path ./safety_benchmarks/counterexamples/OCTMNIST_FC/OCTMNIST_$count.txt --csv_name instances_$count.csv
    ((count++))
done