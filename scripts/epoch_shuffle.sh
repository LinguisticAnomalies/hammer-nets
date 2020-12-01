#!/bin/bash
format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }

 for layer in 0 1 2 3 4 5 6 7 8 9 10 11
    do 
        echo "Shuffling layer $layer"
        for share in 25 50 100
            do
                echo "shuffling $share% attention heads"
                for batch in 1 2 3 4 5
                    do
                        echo "batch $batch"
                        python -W ignore shuffle_layers.py --layer $layer --share $share --eval diff --batch $batch
                    done
            done
    done


for layer in 0 1 2 3 4 5 6 7 8 9 10 11
    do 
        echo "Shuffling layer $layer"
        for share in 25 50 100
            do
                echo "shuffling $share% attention heads"
                for batch in 1 2 3 4 5
                    do
                        echo "batch $batch"
                        python -W ignore shuffle_layers.py --layer $layer --share $share --eval ratio --batch $batch
                    done
            done
    done


for layer in 0 1 2 3 4 5 6 7 8 9 10 11
    do 
        echo "Shuffling layer $layer"
        for share in 25 50 100
            do
                echo "shuffling $share% attention heads"
                for batch in 1 2 3 4 5
                    do
                        echo "batch $batch"
                        python -W ignore shuffle_layers.py --layer $layer --share $share --eval log --batch $batch
                    done
            done
    done
echo "Script completed in $(format_time $SECONDS)"