# Whole Dataset Performance

# ===================================================Coarsen======================================================
# VN
# cora 0.5%
python train_coarsen.py -D cora -R 0.01 -M variation_neighborhoods
# cora 1.3%
python train_coarsen.py -D cora -R 0.03 -M variation_neighborhoods
# cora 2.6%
python train_coarsen.py -D cora -R 0.07 -M variation_neighborhoods
# =========================================================
# citeceer 0.36%
python train_coarsen.py -D citeseer -R 0.01 -M variation_neighborhoods
# citeceer 0.9%
python train_coarsen.py -D citeseer -R 0.04 -M variation_neighborhoods
# citeceer 1.8%
python train_coarsen.py -D citeseer -R 0.1 -M variation_neighborhoods
# =========================================================
#  flickr 0.1%
python train_coarsen.py -D flickr -R 0.001 -M variation_neighborhoods
#  flickr 0.5%
python train_coarsen.py -D flickr -R 0.005 -M variation_neighborhoods
#  flickr 1%
python train_coarsen.py -D flickr -R 0.01 -M variation_neighborhoods
# =========================================================
#  reddit 0.05%
python train_coarsen.py -D reddit -R 0.0005 -M variation_neighborhoods
#  reddit 0.1%

#  reddit 0.2%


# =================================================Condensation===================================================
# DosCond
python train_gcond.py -D cora -S trans -R 1 -H 256 -E 10 --lr 0.01 --wd 5e-4 --nlayers 2 --lr_adj 1e-2 --lr_feat 1e-2 --dropout 0 --eps 5000 --dis_metric mse --one_step


