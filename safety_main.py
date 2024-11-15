#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import argparse
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument("CATEGORY", type=str)
parser.add_argument("ONNX_FILE", type=str, default=None, help='ONNX_FILE')
parser.add_argument("VNNLIB_FILE", type=str, default=None, help='VNNLIB_FILE')
parser.add_argument("RESULTS_FILE", type=str, default=None, help='RESULTS_FILE')
parser.add_argument("TIMEOUT", type=float, default=180, help='timeout for one property')
parser.add_argument("--DEBUG", action='store_true', help='whether to run in debug mode (checking saved adv example)')
parser.add_argument("--NOPGD", action='store_true', help='do not use pdg attack')
parser.add_argument("--TRY_CROWN", action='store_true', help='overwrite bound-prop-method to CROWN to save memory')

args = parser.parse_args()

python_path = sys.executable
library_path = os.path.dirname(os.path.realpath(__file__))

cmd = f"{python_path} {library_path}/../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config {library_path}"


# safetyartist categories
if args.CATEGORY == "smote":
    cmd += "/safety_configs/smote.yaml"

elif args.CATEGORY == "smote_data_aug": 
    cmd += "/safety_configs/smote_data_aug.yaml"

elif args.CATEGORY == "baseline": 
    cmd += "/safety_configs/baseline.yaml"

elif args.CATEGORY == "umce": 
    cmd += "/safety_configs/umce.yaml"

elif args.CATEGORY == "acasxu": # left for testing
    cmd += "/safety_configs/acasxu.yaml"

# elif args.CATEGORY == "test":
  #  pass

else:
    exit("CATEGORY {} not supported yet".format(args.CATEGORY))

# test case may run in other args.CATEGORY at the end of them, so we parse them here to allow correct measurement of overhead.
if os.path.split(args.VNNLIB_FILE)[-1] in ['test_' + f + '.vnnlib' for f in ['nano', 'tiny', 'small']]:
    cmd = f"{python_path} {library_path}/../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config {library_path}/../abcrown_safety/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp21/test.yaml"
elif 'test_prop' in args.VNNLIB_FILE:
    cmd = f"{python_path} {library_path}//../abcrown_safety/alpha-beta-CROWN/complete_verifier/abcrown.py --config {library_path}/../abcrown_safety/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/acasxu.yaml"


cmd += " --precompile_jit"
cmd += " --onnx_path " + str(args.ONNX_FILE)
cmd += " --vnnlib_path " + str(args.VNNLIB_FILE)
cmd += " --results_file " + str(args.RESULTS_FILE)
cmd += " --timeout " + str(args.TIMEOUT)

# save adv example to args.RESULTS_FILE
cmd += " --save_adv_example"

# use CROWN bound propagation, when original run triggers OOM, run_instance.sh will add this flag
if args.TRY_CROWN:
    # This also disables the use of output constraints. They are only useful for alpha-CROWN
    cmd += " --bound_prop_method crown --apply_output_constraints_to"

# verify the adv example everytime it's saved
if args.DEBUG:
    cmd += " --eval_adv_example"

# do not use pdg attack during verification
if args.NOPGD:
    cmd += " --pgd_order=skip"

print("\n------------------------- COMMAND ------------------------------")
print(cmd)
print("----------------------------------------------------------------\n")

ret = os.system(cmd)
if ret != 0:
    # avoid original return code to be > 255, reserve its non-zero feature
    sys.exit(int(ret) % 255 + 1)