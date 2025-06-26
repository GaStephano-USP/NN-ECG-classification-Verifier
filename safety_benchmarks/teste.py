import os
import sys

cmd = "python3 ../../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config ../safety_configs/baseline.yaml --model baseline_model"


print("\n------------------------- COMMAND ------------------------------")
print(cmd)
print("----------------------------------------------------------------\n")

ret = os.system(cmd)
if ret != 0:
    # avoid original return code to be > 255, reserve its non-zero feature
    sys.exit(int(ret) % 255 + 1)