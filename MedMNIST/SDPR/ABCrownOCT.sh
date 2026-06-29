#!/bin/bash
COUNT=0
EPSILON=0.00
LIMIT=784
K=0
SEED=678190
P0x=0
P0y=0
MODE='SnP'
ANGLE=0
OUTPUT_FILE="results/outputs/OCTMNIST/FC/REFEITO_resultadosoctmnist_SnP_0_seed2.txt"
> "$OUTPUT_FILE"
start=`date +%s`
if [ "$MODE" == 'rel_abs' ]; then
    while [ "$(bc <<< "$EPSILON < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --epsilon $EPSILON --mode "rel" --model FC --model_path ./trained_models/OCT_FC_Net/OCT_FC_Net.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        EPSILON="$(bc <<< "$EPSILON + 0.001")"
        echo $EPSILON
    done
elif [ "$MODE" == "SnP" ]; then

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --k $K --mode "SnP" --seed $SEED --p 0 --model FC --model_path  ./trained_models/OCT_FC_Net/OCT_FC_Net.pth #./trained_models/OCT_ConvNet/OCTMNISTCNN3_Max.pth 
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
        echo $K
    done

elif [ "$MODE" == "Rot" ]; then

    while [ "$(bc <<< "$ANGLE < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --angle $ANGLE --mode "Rot" --model CNN --model_path ./trained_models/OCT_ConvNet/OCTMNISTCNN3_Max.pth
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST_CNN.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        ANGLE="$(bc <<< "$ANGLE + 4")"
        echo $ANGLE  
    done

elif [ "$MODE" == "Crop" ]; then
    while [ "$(bc <<< "$P0y < 26")" == "1" ]; do
        P0x=0
        while [ "$(bc <<< "$P0x < 26")" == "1" ]; do
            python3 ./MedMNIST/SDPR/VNNLIBmakerOCT.py --mode "Crop" --P0 $P0x $P0y --altura 3 --largura 3 --model CNN --model_path trained_models/OCT_ConvNet/OCTMNISTCNN3_Max.pth
            output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/OCTMNIST_CNN.yaml)
            match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
            echo "$match" >> "$OUTPUT_FILE"
            P0x="$(bc <<< "$P0x + 1")"  
        done
        P0y="$(bc <<< "$P0y + 1")"
    done    
fi
end=`date +%s`
runtime=$((end-start))
echo "$runtime" >> "$OUTPUT_FILE"