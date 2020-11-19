#!/bin/bash
format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }
 
 for layer in 0 1 2 3 4 5 6 7 8 9 10 11
    do 
        echo "zeroing layer $layer"
        for share in 25 50 100
            do
                echo "zeroing $share% attention heads"
                python shuffle_layers.py --layer $layer --share $share --batch 1
            done
    done
echo "Script completed in $(format_time $SECONDS)"