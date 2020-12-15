#!/bin/bash
format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }
for style in "onetime" "accumu" "comb"
    do
        for share in 25 50 75 100
            do
            echo ""
            echo "Zeroing" $share"% attn heads with style "$style
            python -W ignore zero_attn_heads.py --style $style --share $share --text yes
            echo ""
            done
    done
echo "Script completed in $(format_time $SECONDS)"