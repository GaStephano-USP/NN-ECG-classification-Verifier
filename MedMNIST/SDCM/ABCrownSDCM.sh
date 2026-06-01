#!/bin/bash
COUNT=0
EPSILON=0.14
LIMIT=784
K=0
SEED=60601
MODE='SnP'
P0x=0
P0y=0
ANGLE=0
OUTPUT_FILE="results/outputs/BreastMNIST/CNN/resultadosbreast_SnP_0.txt"
> "$OUTPUT_FILE"
start=`date +%s`
if [ "$MODE" == 'rel_abs' ]; then

    while [ "$(bc <<< "$EPSILON < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDCM/VNNLIBmakerBreastMNIST.py --epsilon $EPSILON --mode "abs"
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/BreastMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        EPSILON="$(bc <<< "$EPSILON + 0.001")"
        echo $EPSILON
    done

elif [ "$MODE" == "SnP" ]; then

    while [ "$(bc <<< "$K < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDCM/VNNLIBmakerBreastMNIST.py --k $K --mode "SnP" --p 0 --seed $SEED --model CNN  --model_path ./trained_models/BreastMNIST/CNN/BreastMNISTCNN3_Max.pth 
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/CNN_BreastMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        K="$(bc <<< "$K + 1")"
        echo $K
    done

elif [ "$MODE" == "Rot" ]; then

    while [ "$(bc <<< "$ANGLE < $LIMIT")" == "1" ]; do
        python3 ./MedMNIST/SDCM/VNNLIBmakerBreastMNIST.py --angle $ANGLE --mode "Rot"
        output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/BreastMNIST.yaml)
        match=$(echo "$output" | grep -Eo '[0-9]+(\.[0-9]+)?%')
        echo "$match" >> "$OUTPUT_FILE"
        ANGLE="$(bc <<< "$ANGLE + 0.5")"
        echo $ANGLE  
    done

elif [ "$MODE" == "Crop" ]; then
    while [ "$(bc <<< "$P0y < 25")" == "1" ]; do
        P0x=0
        while [ "$(bc <<< "$P0x < 25")" == "1" ]; do
            python3 ./MedMNIST/SDCM/VNNLIBmakerBreastMNIST.py --mode "Crop" --P0 $P0x $P0y --altura 3 --largura 3
            output=$(python3 ../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ./safety_configs/BreastMNIST.yaml)
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