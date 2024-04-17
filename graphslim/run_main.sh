# Whole Dataset Performance

# Coarsen
# VN
# cora 5.2%
python train_coarsen.py -D cora -S trans -R 1 -M variation_neighborhoods
# cora 2.6%
python train_coarsen.py -D cora -S trans -R 0.055 -M variation_neighborhoods
# cora 1.3%
python train_coarsen.py -D cora -S trans -R 0.02 -M variation_neighborhoods

# Condensation
# GCond (DosCond)
python train_gcond.py -D cora -S trans -R 1 -H 256 -E 10 --lr 0.01 --wd 5e-4 --nlayers 2 --lr_adj 1e-2 --lr_feat 1e-2 --dropout 0 --eps 5000 --dis_metric mse --one_step


