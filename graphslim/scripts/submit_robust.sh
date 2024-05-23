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

for att in $attacks; do
  for dataset in "${!datasets[@]}"; do
    for method in $methods; do
      for r in ${datasets[$dataset]}; do
        sbatch all.sh -D $dataset -M $method $r $att ${ptbr[$att]}
      done
    done
  done
done