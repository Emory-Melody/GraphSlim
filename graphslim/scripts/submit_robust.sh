declare -A datasets
declare -A ptbr
datasets["cora"]="0.5"
datasets["citeseer"]="0.5"
datasets["flickr"]="0.01"
ptbr["random_adj"]="0.5"
ptbr["metattack"]="0.25"
methods="random gcond gcondx kcenter geom"
attacks="random_adj random_feat metattack"
seeds="1 2 3"

for att in $attacks; do
  for dataset in "${!datasets[@]}"; do
    for method in $methods; do
      for r in ${datasets[$dataset]}; do
        for seed in $seeds; do
          sbatch robust.sh $dataset $method $r $att ${ptbr[$att]} $seed
        done
      done
    done
  done
done