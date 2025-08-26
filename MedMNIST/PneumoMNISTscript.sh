#!/bin/bash
count=0
python3 ./MedMNIST/VNNLIBmakerPneumonia.py
while [ $count -le 389 ]; do
    python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/FC_pneumoniaMNIST.yaml --model PneumoniaMNIST --cex_path ./safety_benchmarks/counterexamples/PneumoniaMNIST/PneumoniaMNIST_$count.txt --csv_name instances_$count.csv
    ((count++))
done