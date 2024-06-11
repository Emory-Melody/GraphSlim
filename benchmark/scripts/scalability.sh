# Scalability
for method in random herding kcenter gcondx geom gcond msgc; do
  for dataset in cora citeseer ogbn-arxiv flickr reddit; do
    case $dataset in
      ogbn-arxiv)
        for r in 0.00045 0.0005 0.001 0.005 0.01 0.025 0.05 0.1; do
          python ../train_all.py -M $method -D $dataset -R $r
        done
        ;;
      reddit)
        for r in 0.00025 0.0005 0.001 0.002 0.005 0.01 0.015 0.02 0.025 0.03; do
          python ../train_all.py -M $method -D $dataset -R $r
        done
        ;;
    esac
  done
done