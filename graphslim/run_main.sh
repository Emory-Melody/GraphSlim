# Whole Dataset Performance

# ===================================================Coarsen======================================================
# VN
# cora 1.3%
python train_coarsen.py -D cora -S trans -R 0.02 -M variation_neighborhoods
# cora 2.6%
python train_coarsen.py -D cora -S trans -R 0.055 -M variation_neighborhoods
# cora 5.2%
python train_coarsen.py -D cora -S trans -R 1 -M variation_neighborhoods
# =========================================================
# citeceer 1.8%
python train_coarsen.py -D citeseer -S trans -R 0.025 -M variation_neighborhoods
# citeceer 3.6%
python train_coarsen.py -D citeseer -S trans -R 1 -M variation_neighborhoods
# =========================================================
#  flickr 0.1%
python train_coarsen.py -D flickr -S ind -R 0.001 -M variation_neighborhoods
#  flickr 0.5%
python train_coarsen.py -D flickr -S ind -R 0.005 -M variation_neighborhoods
#  flickr 1%
python train_coarsen.py -D flickr -S ind -R 0.01 -M variation_neighborhoods
# =========================================================
#  reddit 0.05%
python train_coarsen.py -D reddit -S ind -R 0.005 -M variation_neighborhoods
#  reddit 0.1%

#  reddit 0.2%


# =================================================Condensation===================================================
# DosCond
python train_gcond.py -D cora -S trans -R 1 -H 256 -E 10 --lr 0.01 --wd 5e-4 --nlayers 2 --lr_adj 1e-2 --lr_feat 1e-2 --dropout 0 --eps 5000 --dis_metric mse --one_step


