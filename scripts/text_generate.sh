format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }
 for share in 25 50 75 100
 do
    for data_type in "full" "mild" "slight" "sev"
    do
        for hammer_style in "comb" "accumu" "onetime"
        do
            for zero_style in "first" "random"
            do
                python -W ignore zero_attn_heads_style.py --hammer_style $hammer_style --zero_style $zero_style --data_type data_type --share $share --text no
            done
        done
    done
done
echo "Script completed in $(format_time $SECONDS)"