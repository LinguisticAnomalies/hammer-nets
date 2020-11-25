#!/bin/bash
format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }
 for share in 25 50 100
    do
    echo "zeroing $share% attention heads"
    for eval in "diff" "ratio" "log"
        do
        echo "evaluation method: $eval"
        for style in "onetime" "accumu" "comb"
            do
            echo "zeroing style: $style"
            python zero_attn_heads.py --style $style --eval $eval  --share $share
            done
        done
    done
echo "Script completed in $(format_time $SECONDS)"