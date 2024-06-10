# Graph Property Preservation
# Must obtain condensed graph of each methods by running performance.sh before running this script.
for method in vng gcond msgc sgdd; do
  for dataset in cora citeseer ogbn-arxiv flickr reddit; do
      python Benchmark/graph_property.py -M $method -D $dataset
  done
done


for method in gcondx geom; do
  for dataset in cora citeseer ogbn-arxiv flickr reddit; do
      python Benchmark/graph_property_no_structure.py -M $method -D $dataset
  done
done


# Whole results
for dataset in cora citeseer ogbn-arxiv flickr reddit; do
    python Benchmark/graph_property.py -M $method -D $dataset
done