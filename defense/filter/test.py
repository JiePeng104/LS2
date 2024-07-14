import os
import sys

os.chdir(sys.path[0])
sys.path.append('../../')
os.getcwd()
import numpy as np
import subprocess
import platform
import defense.filter.quantum as quantum



# pca = np.load('../../record/badnet_0_1/quantum/mask-pca-target.npy')
# kmeans = np.load('../../record/badnet_0_1/quantum/mask-kmeans-target.npy')
# quantum = np.load('../../record/badnet_0_1/quantum/mask-rcov-target.npy')
# target = np.load('../../record/badnet_0_1/quantum/label_0-poison_indicator.npy')

work_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
current_path = os.path.dirname(__file__)
# # cmd = ['julia', '--project=.', f'{current_path}/test.jl', f'{work_path}/record/badnet_0_1/quantum*label_0-out_array.npy']
cmd = f'julia --project=. {current_path}/test.jl {work_path}/record/badnet_0_1/quantum*label_0-out_array.npy'
#
# print(cmd)
# p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
# for line in p.stdout:
#     print(line)
# p.wait()
# from defense.filter.quantum import filter_comparator
#
# filter_info = [{"name": "pca", "poison_indicator": "mask-pca-target.npy"},
#                {"name": "k-means", "poison_indicator": "mask-kmeans-target.npy"},
#                {"name": "quantum", "poison_indicator": "mask-rcov-target.npy"}]
# filter_comparator(f"{work_path}/record/badnet_0_1/quantum", 'label_0-poison_indicator.npy', filter_info)

# quantum.quantum_filter('/mnt/BackdoorBench-main/record/badnet_0_1/quantum', 'label_0-out_array_trans.npy')

# print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# print(os.path.dirname(__file__))
# print(os.path.join(os.path.dirname(__file__), 'test.jl'))


all_kpca= f"{work_path}/record/badnet_0_1/quantum/label_0-all_kpca_gamma_10_ncom-100-out_array_trans.npy"
kpca= f"{work_path}/record/badnet_0_1/quantum/label_0-kpca_gamma_10-out_array_trans.npy"
all_kpca_output_array = np.load(all_kpca)
kpca_output_array = np.load(kpca)
print('hi')



