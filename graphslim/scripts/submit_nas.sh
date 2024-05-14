declare -A datasets
datasets["cora"]="0.5"
datasets["ogbn-arxiv"]="0.01"
datasets["reddit"]="0.002"

methods="gcondx geom sfgc doscond gcond msgc gcsntk sgdd cent_d cent_p random herding kcenter averaging clustering variation_neighborhoods vng"

for dataset in "${!datasets[@]}"; do
    for method in $methods; do
        for r in ${datasets[$dataset]}; do
            sbatch transferbility.sh $dataset $method $r
        done
    done
done