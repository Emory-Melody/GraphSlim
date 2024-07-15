# Performance
# For geom, we obtain its performance by the source code of the authors but condense graph by our package for other experiments except performance.
# For sfgc, we obtain its performance by the source code of the authors.
for method in random herding kcenter averaging gcondx vng gcondx geom sfgc gcsntk doscond gcond msgc sgdd; do
  for dataset in cora citeseer flickr; do
    case $dataset in
      cora)
        for r in 0.1 0.25 0.5; do
          python ../train_all.py -M $method -D $dataset -R $r
        done
        ;;
      ogbn-arxiv)
        for r in 0.001 0.005 0.01; do
          python ../train_all.py -M $method -D $dataset -R $r
        done
        ;;
      flickr)
        for r in 0.001 0.005 0.01; do
          python ../train_all.py -M $method -D $dataset -R $r
        done
        ;;
      reddit)
        for r in 0.0005 0.001 0.002; do
          python ../train_all.py -M $method -D $dataset -R $r
        done
        ;;
    esac
  done
done

# Whole results
for dataset in cora citeseer ogbn-arxiv flickr reddit; do
  python ../run_eval.py -D $dataset -W
done