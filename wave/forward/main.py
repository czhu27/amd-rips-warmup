import sys
import datetime
# TODO: Oh boy... these paths are a mess. oh well!
sys.path.append(sys.path[0] + "/../..")
from data import process_wave_data
sys.path.append(sys.path[0] + "/app")
import yaml
from simulator import Simulator
import os

def main(params):
    # Create simulator and run
    simulator = Simulator(params)
    simulator.run()
    simulator.finalize()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", type=str, default=None)

    args = parser.parse_args()
    if args.configs is None:
        raise ValueError("Specify which config to use.")

    # Create all parameters
    params = yaml.safe_load(open(args.configs))

    fname = os.path.basename(args.configs).split(".")[0]
    params['data_dir'] = params['data_dir'] #+ "/" + fname
    
    main(params)