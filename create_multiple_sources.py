import os
import argparse
from shutil import copyfile
import yaml
import datetime
import json

def source_to_str(source_pair):
    x = str(source_pair[0])
    y = str(source_pair[1])

    return x.replace('.', '') + '-' + y.replace('.', '')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sources", type=json.loads)
    args = parser.parse_args()

    random_sources_params = 'wave/params/multiple_sources/'
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S/")
    os.makedirs(random_sources_params + run_name, exist_ok=True)

    default_file_path = 'wave/params/defaults.yaml'
    
    for source_list in args.sources:
        file_name = random_sources_params + run_name + source_to_str(source_list) + '.yaml'
        copyfile(default_file_path, file_name)
        with open(file_name, 'r') as f:
            file_dict = yaml.safe_load(f)
        
        file_dict['src_loc'] = [source_list]
        file_dict['data_dir'] = 'data/wave/sources_' + run_name + '/' + source_to_str(source_list) + '/'
        
        with open(file_name, 'w') as f:
            file_dict = yaml.safe_dump(file_dict, f, default_flow_style=False)

