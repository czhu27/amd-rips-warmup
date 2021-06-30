import os
import sys
import yaml

# Usage: python3 errorsearch.py extErrorBound intErrorBound
# extErrorBound = Extrapolation error upper bound
# intErrorBound = Interpolation error upper bound

def find(name, path):
    res = []
    for root, dirs, files in os.walk(path):
        if name in files:
            res.append(os.path.join(root, name))
    return res

# Find all errors.yaml files in output folder
errorPaths = find("results.yaml", "./output/no_exp_id/")
goodruns = []
for path in errorPaths:
    run = path.split("/")[3]
    with open(path) as f:
        errorDict = yaml.safe_load(f)
        # Compare extrapolation and interpolation error to arguements
        if float(errorDict['error_ext']) <= float(sys.argv[1]) and float(errorDict['error_int']) <= float(sys.argv[2]):
            goodruns.append(run)

# Prints run numbers for runs that have lower errror numbers
for run in goodruns:
    print(run)