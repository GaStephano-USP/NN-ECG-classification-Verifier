#!/bin/bash

LIMIT=784
K=0
MODE="SnP"
SEED=1950
NETWORK=0  

while [ "$NETWORK" == "0" ]; do
    echo "PneumoniaMNIST com avgpool e proporção 0%"
    OUTPUT_FILE="results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_SnP_avg_0.txt"
    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        echo $K
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --k $K --mode "SnP" --seed $SEED --p 0 --model CNN --model_path trained_models/PneumoniaMNIST/CNN/PneumoniaMNISTCNN3_avg.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/CNN_avg_pneumoniaMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
    done
    K=0
    NETWORK="$(bc <<< "$NETWORk + 1")"
done

while [ "$NETWORK" == "1" ]; do
    echo "PneumoniaMNIST com avgpool e proporção 100%"
    OUTPUT_FILE="results/outputs/PneumoniaMNIST/CNN/resultadospneumomnist_SnP_avg_100.txt"

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        echo $K
        python3 ./MedMNIST/SDP/VNNLIBmakerPneumonia.py --k $K --mode "SnP" --seed $SEED --p 100 --model CNN --model_path trained_models/PneumoniaMNIST/CNN/PneumoniaMNISTCNN3_avg.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/CNN_avg_pneumoniaMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
    done
    K=0
    NETWORK="$(bc <<< "$NETWORk + 1")"
done

while [ "$NETWORK" == "2" ]; do
    echo "OCTMNIST com maxpool e proporção 0%"
    OUTPUT_FILE="results/outputs/OCTMNIST/CNN/resultadosoct_SnP_max_0.txt"

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --k $K --mode "SnP" --seed $SEED --p 0 --model CNN --model_path trained_models/OCT_ConvNet/OCTMNISTCNN3_Max.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST_CNN.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
        echo $K
    done
    K=0
    NETWORK="$(bc <<< "$NETWORk + 1")"
done

while [ "$NETWORK" == "3" ]; do
    echo "OCTMNIST com maxpool e proporção 100%"
    OUTPUT_FILE="results/outputs/OCTMNIST/CNN/resultadosoct_SnP_max_100.txt"

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --k $K --mode "SnP" --seed $SEED --p 100 --model CNN --model_path trained_models/OCT_ConvNet/OCTMNISTCNN3_Max.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST_CNN.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
        echo $K
    done
    K=0
    NETWORK="$(bc <<< "$NETWORk + 1")"
done