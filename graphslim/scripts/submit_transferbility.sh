declare -A datasets
datasets["cora"]="0.5"
datasets["ogbn-arxiv"]="0.01"
datasets["reddit"]="0.002"

methods="gcondx geom sfgc doscond gcond msgc sgdd random kcenter averaging vng"

for dataset in "${!datasets[@]}"; do
    for method in $methods; do
        sbatch transferbility.sh $dataset $method
    done
done