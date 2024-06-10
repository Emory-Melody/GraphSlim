# Neural Architecture Search
# Must obtain condensed graph of each methods by running performance.sh before running this script.
for method in random kcenter gcondx geom sfgc doscond gcond msgc; do
  for dataset in cora ogbn-arxiv reddit; do
      python Benchmark/run_nas.py -M $method -D $dataset
  done
done
