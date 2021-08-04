#!/bin/bash

cd output/wave/single
echo "WO WO WEE WA"
array=(`find -mindepth 2 -maxdepth 2 -type d`)
for i in "${array[@]}" ; do
    cd $i
    resArray=(`find -type f -name results.yaml`)
    if [ "${#resArray[@]}" -eq "0" ] ; then
        cd ../../
        rm -rf $i
    else
        cd ../../
    fi
done