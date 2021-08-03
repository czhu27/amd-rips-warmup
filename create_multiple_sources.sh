#!/bin/bash

# add sources as a 2D-List i.e. [[0.1, 0.1], [0.2, 0.2], [0.3,0.3]]
python3 create_multiple_sources.py -s '[[0.2, 0.2], [0.1, 0.1]]'
array=(`find wave/params/multiple_sources/ -type d  | sort -r | head -2`)
find "${array[0]}" -type f -print0 | while IFS= read -r -d $'\0' file; 
   do python3 wave/forward/main.py -c $file ;
done