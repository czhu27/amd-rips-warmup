import os
import sys
import yaml

def find(name, path):
    res = []
    for root, dirs, files in os.walk(path):
        if name in files:
            res.append(os.path.join(root, name))
    return res

def avg(lst):
    return sum(lst)/len(lst)

def create_run_summary_wave(run_path):
    '''
    Create a text file containing averages for errors across all
    trials in a run
    '''
    errorPaths = find("results.yaml", run_path)
    error_int = []
    error_ext = []
    training_time = []
    for path in errorPaths:
        with open(path) as f:
            errorDict = yaml.safe_load(f)
            error_int.append(float(errorDict["interpolation error (t <= 1)"]))
            error_ext.append(float(errorDict["extrapolation error (1 < t)"]))
            training_time.append(float(errorDict["training_time"][:-2]))
    error_int_avg = avg(error_int)
    error_ext_avg = avg(error_ext)
    training_time_avg = avg(training_time)

    with open(run_path + "/summary.txt", 'w') as f:
        f.write("Avg Interpolation Error (t <= 1): " + str(error_int_avg) + "\n")
        f.write("Avg Extrapolation Error (t > 1): " + str(error_ext_avg) + "\n")
        f.write("Avg Training Time: " + str(training_time_avg) + "\n")

def create_run_summary(run_path):
    '''
    Create a text file containing averages for errors across all
    trials in a run
    '''
    errorPaths = find("results.yaml", run_path)
    error1s = []
    error2s = []
    error3s = []
    for path in errorPaths:
        with open(path) as f:
            errorDict = yaml.safe_load(f)
            error1s.append(float(errorDict['interpolation error (1x1 square)']))
            error2s.append(float(errorDict['extrapolation error (2x2 ring)']))
            error3s.append(float(errorDict['extrapolation error (3x3 ring)']))
    
    error1avg = avg(error1s)
    error2avg = avg(error2s)
    error3avg = avg(error3s)

    with open(run_path + "/summary.txt", 'w') as f:
        f.write("Avg Interpolation Error [-1, 1] x [-1, 1]: " + str(error1avg) + "\n")
        f.write("Avg Extrapolation Error [-2, 2] x [-2, 2]: " + str(error2avg) + "\n")
        f.write("Avg Extrapolation Error [-3, 3] x [-3, 3]: " + str(error3avg) + "\n")


def create_overall_summary(main_path):
    '''
    Create a text file listing averarge error information for each 
    run in a grid search
    '''
    summaryPaths = find("summary.txt", main_path)
    descriptor = main_path.split("/")[-1]

    with open(main_path + "/" + descriptor + "_summary.txt", 'w') as f1:
        for path in summaryPaths:
            run = path.split("/")[4]
            f1.write(run + " summary\n")
            with open(path, 'r') as f2:
                for line in f2:
                    f1.write(line)
            f1.write("\n")