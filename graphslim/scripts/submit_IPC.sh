reduction_rate="0.001 0.005 0.01 0.05 0.1 0.25"

dataset="ogbn-arxiv"
methods="random herding kcenter gcondx geom gcond msgc"

for method in $methods; do
    for r in $reduction_rate; do
        sbatch main.sh $dataset $method $r
    done
done