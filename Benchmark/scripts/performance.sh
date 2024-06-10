# Performance
for method in random kcenter gcondx geom gcond; do
  for dataset in cora citeseer flickr; do
    case $dataset in
      cora)
        for r in 0.1 0.25 0.5; do
          python Benchmark/train_all.py -M $method -D $dataset -R $r
        done
        ;;
      ogbn-arxiv)
        for r in 0.001 0.005 0.01; do
          python Benchmark/train_all.py -M $method -D $dataset -R $r
        done
        ;;
      flickr)
        for r in 0.001 0.005 0.01; do
          python Benchmark/train_all.py -M $method -D $dataset -R $r
        done
        ;;
      reddit)
        for r in 0.0005 0.001 0.002; do
          python Benchmark/train_all.py -M $method -D $dataset -R $r
        done
        ;;
    esac
  done
done

# Whole results
for dataset in cora citeseer ogbn-arxiv flickr reddit; do
  python Benchmark/run_eval.py -D $dataset -W
done