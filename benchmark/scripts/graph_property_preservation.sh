# Graph Property Preservation
# Must obtain condensed graph of each methods by running performance.sh before running this script.
for dataset in cora citeseer ogbn-arxiv flickr reddit; do
    python ../graph_property.py -D $dataset -W
done


for dataset in cora citeseer ogbn-arxiv flickr reddit; do
    python ../graph_property_no_structure.py -D $dataset -W
done