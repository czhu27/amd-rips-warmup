import os
import sys
import argparse
from shutil import rmtree
import yaml
import glob
from PIL import Image

sys.path.append(sys.path[0] + "/wave/forward/app")
from simulator import Simulator

def create_gif(path, time_duration):
    os.chdir(path)
    os.chdir('figs')
    imgs = [img for img in glob.glob('*.png')]
    imgs = sorted(imgs, key = lambda fpath : int(fpath.split("_")[-1][:-4]))
    frames = [Image.open(img) for img in imgs]

    os.makedirs('../../../../gifs', exist_ok=True)
    frames[0].save('../../../../gifs/' + path.split('/')[-2] + '-' + path.split('/')[-1] + '.gif', save_all=True, 
        append_images=frames[1:], loop=0, duration=time_duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default=None)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-s", "--step_size", type=int, default=5)
    parser.add_argument("-t", "--time_duration", type=int, default=5)
    args = parser.parse_args()

    if args.data_dir is not None:
        create_gif(args.data_dir, args.time_duration)
    elif args.config is not None:
        try:
            with open('wave/params/' + args.config + '.yaml', 'r') as f:
                params = yaml.safe_load(f)
            params["display_img"] = args.step_size
            params["data_dir"] = params["data_dir"] + args.config

            simulator = Simulator(params)
            simulator.run()
            simulator.finalize()

            create_gif(params["data_dir"], args.time_duration)
            os.chdir('../../../../../')
            rmtree(params["data_dir"])
        except:
            print("File not found")
    else:
        raise ValueError("Specify a data_directory or param file to create the simulation gif from")
