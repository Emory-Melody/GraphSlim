# Robustness
for method in random kcenter gcondx geom gcond; do
  for dataset in cora citeseer flickr; do
    for attack in metattack random_adj random_feat; do
     python Benchmark/train_all.py -M $method -D $dataset -A $attack -P 0.5
    done
  done
done

# Whole results
for dataset in cora citeseer flickr; do
  for attack in metattack random_adj random_feat; do
   python Benchmark/train_all.py -M $method -D $dataset -A $attack -P 0.5 -W
  done
done