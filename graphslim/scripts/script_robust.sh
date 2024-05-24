declare -A datasets
declare -A ptbr
datasets["cora"]="0.1 0.5"
datasets["citeseer"]="0.1 0.5"
datasets["flickr"]="0.001 0.01"
ptbr["random"]="0.5"
ptbr["random_feat"]="0.5"
ptbr["metattack"]="0.25"
methods="gcond gcondx geom kcenter"
attacks="random random_feat metattack"
gpus=(0 1 2 3)

gpu_index=0

for att in $attacks; do
  for dataset in "${!datasets[@]}"; do
    for method in $methods; do
      for r in ${datasets[$dataset]}; do
        gpu_id=${gpus[$gpu_index]}
        python train_all.py -D $dataset -M $method -R $r -P ${ptbr[$att]} -A $att -G $gpu_id
        gpu_index=$(( (gpu_index + 1) % 4 ))
      done
    done
  done
done

