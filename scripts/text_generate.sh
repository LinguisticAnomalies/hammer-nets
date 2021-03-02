format_time() {
  ((h=${1}/3600))
  ((m=(${1}%3600)/60))
  ((s=${1}%60))
  printf "%02d:%02d:%02d\n" $h $m $s
 }
 for share in 25 50 75 100
 do
    for hammer_style in "comb" "accumu" "onetime"
    do
        for zero_style in "first" "random"
        do
            echo $share $data_type $hammer_style $zero_style
            python -W ignore zero_attn_heads_style.py --hammer_style $hammer_style --zero_style $zero_style --data_type ccc --share $share --text no
        done
    done
done
echo "Script completed in $(format_time $SECONDS)"