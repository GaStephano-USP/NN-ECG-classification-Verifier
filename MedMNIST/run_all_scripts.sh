#!/bin/bash
LIMIT=784
SEED=60601

echo "OCTMNIST proporção 100%"
OUTPUT_FILE="results/outputs/OCTMNIST/FC/resultadososctmnist_SnP100.txt"
K=0
while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
    echo $K
    python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --k $K --mode "SnP" --seed $SEED --p 100 --model FC --model_path ./trained_models/OCT_FC_Net/OCT_FC_Net.pth
    output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST.yaml)
    match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
    echo "$match" >> "$OUTPUT_FILE"
    K="$(bc <<< "$K + 1")"
done

echo "OCTMNIST proporção 50%"
OUTPUT_FILE="results/outputs/OCTMNIST/FC/resultadosoctmnist_SnP50.txt"
K=0
while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
    echo $K
    python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --k $K --mode "SnP" --seed $SEED --p 50 --model FC --model_path ./trained_models/OCT_FC_Net/OCT_FC_Net.pth
    output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST.yaml)
    match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
    echo "$match" >> "$OUTPUT_FILE"
    K="$(bc <<< "$K + 1")"
done

echo "OCTMNIST proporção 0%"
OUTPUT_FILE="results/outputs/OCTMNIST/FC/resultadosoctmnist_SnP0.txt"
K=500
LIMIT_0=700
while [ "$(bc <<< "$K < $LIMIT_0")" == "1" ]; do
    echo $K
    python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --k $K --mode "SnP" --seed $SEED --p 0 --model FC --model_path ./trained_models/OCT_FC_Net/OCT_FC_Net.pth
    output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST.yaml)
    match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
    echo "$match" >> "$OUTPUT_FILE"
    K="$(bc <<< "$K + 1")"
done

echo "fim"