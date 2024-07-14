python ./attack/blended_adap_ls.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name blended_adap_pre_p-0.003_cover-0.003 --cover_rate 0.003
python ./attack/blended_adap_ls.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name blended_adap_LS_sr-0.1_p-0.003_cover-0.003 --cover_rate 0.003 --smooth_rate 0.1
python ./attack/blended_adap_ls.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name blended_adap_LS2_pre_p-0.003_cover-0.003 --cover_rate 0.003 --dynamic_ls True --warmup_end 10 --p_rate 0.2 --n_rate 0.0 --cover_rate 0.003 --log_aum True --smooth_rate 0.1


python ./attack/adap_patch_ls.py --yaml_path ../config/attack/adap_patch/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name adap_patch_pre_p-0.003_cover-0.003 --cover_rate 0.003
python ./attack/adap_patch_ls.py --yaml_path ../config/attack/adap_patch/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name adap_patch_LS_sr-0.1_p-0.003_cover-0.003 --cover_rate 0.003 --smooth_rate 0.1
python ./attack/adap_patch_ls.py --yaml_path ../config/attack/adap_patch/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name adap_patch_LS2_p-0.003_prate-0.2_nrate-0_wepoch-10_cover-0.003 --dynamic_ls True --warmup_end 10 --p_rate 0.2 --n_rate 0.0 --cover_rate 0.003 --log_aum True --smooth_rate 0.1
