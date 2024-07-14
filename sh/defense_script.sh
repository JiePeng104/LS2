python ./defense/anp/anp.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/anp/cifar10.yaml --dataset cifar10
python ./defense/awm/awm.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/awm/cifar10.yaml --dataset cifar10
python ./defense/nc/nc.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/nc/cifar10.yaml --dataset cifar10
python ./defense/i-bau/i-bau.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/i-bau/cifar10.yaml --dataset cifar10
python ./defense/abl/abl.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/abl/cifar10.yaml --dataset cifar10

python ./latent_separation_defense/adap_patch_filter.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --load_path blended_adap_LS2_pre_p-0.003_cover-0.003 --save_folder_name blended_adap_LS2_pre_p-0.003_cover-0.003 --tsne_visual True




