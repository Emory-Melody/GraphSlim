declare -A datasets
datasets["cora"]="0.1 0.5 1"
datasets["citeseer"]="0.1 0.5 1"
datasets["ogbn-arxiv"]="0.001 0.005 0.01"
datasets["flickr"]="0.001 0.005 0.01"
datasets["reddit"]="0.0005 0.001 0.002"

methods="gcondx gcond doscond sgdd kcenter herding random cent_p cent_d averaging variation_neighborhoods clustering vng"

for dataset in "${!datasets[@]}"; do
    for method in $methods; do
        for r in ${datasets[$dataset]}; do
            sbatch main.sh $dataset $method $r
        done
    done
done