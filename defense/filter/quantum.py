import logging
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import subprocess
import platform
from numpy.linalg import norm
from tqdm import tqdm
sys.path.append('../')
sys.path.append(os.getcwd())
import defense.filter.scan as scan


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--save_path", type=str)
    return parser


def scatter_visualization(prefix, save_path, num_class, output_array, target_array):
    plt.rcParams['figure.figsize'] = 10, 10
    plt.figure()
    point_list = [[] for i in range(num_class + 1)]
    colors = ["#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590",
              "#277da1"]
    for i in range(len(target_array[:, 0])):
        t = target_array[i, 0]
        point = output_array[i, :]
        point_list[t].append(point)
    poison_rep = np.array(point_list[num_class])
    plt.scatter(poison_rep[:, 0], poison_rep[:, 1], c='#161a1d', marker='x', label=num_class, alpha=0.1,
                edgecolors='none')
    for i in range(num_class):
        representation = np.array(point_list[i])
        plt.scatter(representation[:, 0], representation[:, 1], c=colors[i], label=i, alpha=0.4,
                    edgecolors='none')
    plt.legend()
    plt.savefig(os.path.join(save_path, prefix + '-tsne.pdf'), bbox_inches='tight')


def scan_filter(args, model, adv_train, benign_test):
    plat = platform.system().lower()
    if plat == 'windows':
        absolute_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # linux
    else:
        absolute_path = os.path.dirname(os.getcwd())
    quantum_path = os.path.join(absolute_path, args.save_path[3:], 'quantum')
    if not os.path.exists(quantum_path):
        os.mkdir(quantum_path)
    random_clean_index = np.random.choice(np.arange(len(benign_test)), 1000, replace=False)
    benign_test.subset(random_clean_index)
    scan_poi_index = scan.cleanser(adv_train, benign_test, model, args.num_classes)
    np.save(os.path.join(quantum_path, 'mask-SCan-target.npy'), scan_poi_index)
    poison_indicator = adv_train.poison_indicator
    filter_indicator = np.zeros(len(poison_indicator), dtype=int)
    filter_indicator[scan_poi_index] = 1
    miss_clean, miss_poi, acc_clean, acc_poi = 0, 0, 0, 0
    for i in range(len(poison_indicator)):
        if poison_indicator[i] == 1:
            if filter_indicator[i]:
                acc_poi += 1
            else:
                miss_poi += 1
        else:
            if filter_indicator[i]:
                miss_clean += 1
            else:
                acc_clean += 1
    print(f"SCan: # acc_poi={acc_poi}, # miss_poi={miss_poi}, # acc_clean={acc_clean}, # miss_clean={miss_clean}")
    logging.info(
        f"SCan: # acc_poi={acc_poi}, # miss_poi={miss_poi}, # acc_clean={acc_clean}, # miss_clean={miss_clean}")


def filters(args, model, adv_train, target_class, device):
    # prepare data loader
    targets_list = adv_train.targets
    target_index = []
    for i in range(len(targets_list)):
        if int(targets_list[i]) == target_class:
            target_index.append(i)
    target_ds = deepcopy(adv_train)
    target_ds.subset(np.array(target_index))
    dl = DataLoader(
        dataset=target_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # get representations
    model.to(device)
    model.eval()

    plat = platform.system().lower()

    # save representations
    # win
    if plat == 'windows':
        absolute_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # linux
    else:
        absolute_path = os.getcwd()
        absolute_path = os.path.dirname(absolute_path)

    quantum_path = os.path.join(absolute_path, args.save_path[3:], 'quantum')
    if not os.path.exists(quantum_path):
        os.mkdir(quantum_path)

    np_file = 'label_' + str(target_class) + '-out_array_trans.npy'

    # load or compute reps
    reps = []
    output_array = []
    if not os.path.exists(os.path.join(quantum_path, np_file)):
        for batch_idx, (inputs, targets, *additional_info) in enumerate(dl):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs, targets
            outputs = model(inputs)
            penultimate_outputs = model.get_penultimate()
            output_np = penultimate_outputs.data.cpu().numpy()
            target_np = targets.data.cpu().numpy()
            reps.append(output_np)
        output_array = np.concatenate(reps, axis=0)
    else:
        output_array = np.load(os.path.join(quantum_path, np_file)).T

    # save sorts of arrays
    np.save(os.path.join(quantum_path, np_file), output_array.T, allow_pickle=False)
    np.save(os.path.join(quantum_path, 'label_' + str(target_class) + "-poison_indicator.npy"),
            target_ds.poison_indicator, allow_pickle=False)
    np.save(os.path.join(quantum_path, 'label_' + str(target_class) + "-original_index.npy"), target_ds.original_index,
            allow_pickle=False)

    num_poi = 0
    for i in target_ds.poison_indicator:
        if i == 1:
            num_poi += 1

    # ori
    origin_reps(output_array, target_ds.poison_indicator, num_poi, target_class, quantum_path, np_file)

    # d_intra, d_inter, r2 = class_separation(output_array, target_ds.poison_indicator)
    # logging.info(f"Class Separation, d_intra:{d_intra}, d_inter:{d_inter}, R^2:{r2}")


def origin_reps(output_array, poi_indicator, num_poi, target_class, quantum_path, np_file):
    # filter defense
    quantum_filter(quantum_path, np_file, num_poi)
    # get mean delta
    delta = mean_delta(output_array, poi_indicator)
    print(f"label_{target_class} L2 norm of mean diff: {delta}")
    logging.info(f"label_{target_class} L2 norm of mean diff: {delta}")
    filter_info = [{"name": "pca", "poison_indicator": "mask-pca-target-ori.npy"},
                   {"name": "k-means", "poison_indicator": "mask-kmeans-target-ori.npy"},
                   {"name": "quantum", "poison_indicator": "mask-rcov-target-ori.npy"}]
    filter_comparator(quantum_path, 'label_' + str(target_class) + "-poison_indicator.npy", filter_info)


def quantum_filter(save_path, np_file, num_poi, suffix='ori'):
    plat = platform.system().lower()
    if plat == 'windows':
        current_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'defense/filter')
    else:
        current_path = os.path.join(os.path.dirname(os.getcwd()), 'defense/filter')
    cmd = f'julia --project=. {current_path}/run_filters.jl {save_path}*{np_file}*{num_poi}*{suffix}'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line)
    p.wait()


def mean_delta(reps_vec, poi_indicator):
    clean_reps = []
    poi_reps = []
    for i in range(len(poi_indicator)):
        reps_i = reps_vec[i]
        if poi_indicator[i] == 0:
            clean_reps.append(reps_i)
        else:
            poi_reps.append(reps_i)
    clean_reps = np.array(clean_reps)
    poi_reps = np.array(poi_reps)
    clean_mean = clean_reps.mean(0)
    poi_mean = poi_reps.mean(0)
    diff = clean_mean - poi_mean
    return diff.dot(diff)


def class_separation(reps_vec, poi_indicator):
    clean_reps = []
    poi_reps = []
    for i in range(len(poi_indicator)):
        reps_i = reps_vec[i]
        if poi_indicator[i] == 0:
            clean_reps.append(reps_i)
        else:
            poi_reps.append(reps_i)

    clean_reps = np.array(clean_reps)
    poi_reps = np.array(poi_reps)

    vec_list = [clean_reps, poi_reps]
    d_intra, d_inter = 0.0, 0.0

    for reps in vec_list:
        length = len(reps)
        d_intra_k = 0.0
        for i in tqdm(range(length)):
            for j in range(length):
                cos_sim = 1-reps[i].dot(reps[j]) / (norm(reps[i]) * norm(reps[j]))
                d_intra_k += cos_sim
        d_intra += d_intra_k / (length**2)
    d_inter = d_intra
    d_intra = d_intra / len(vec_list)

    length_j = len(clean_reps)
    length_k = len(poi_reps)
    d_inter_jk = 0.0
    for m in tqdm(range(length_j)):
        for n in range(length_k):
            cos_sim = 1 - clean_reps[m].dot(poi_reps[n]) / (norm(clean_reps[m]) * norm(poi_reps[n]))
            d_inter_jk += cos_sim
    d_inter += ((d_inter_jk / (length_j*length_k)) * 2)
    d_inter = d_inter / (len(vec_list)**2)
    r2 = 1 - (d_intra / d_inter)
    return d_intra, d_inter, r2


def filter_comparator(save_path, t_pi, filters_pi):
    poison_indicator = np.load(os.path.join(save_path, t_pi))
    for info in filters_pi:
        miss_clean, miss_poi, acc_clean, acc_poi = 0, 0, 0, 0
        name = info["name"]
        filter_pi = np.load(os.path.join(save_path, info["poison_indicator"]))
        for i in range(len(poison_indicator)):
            if poison_indicator[i] == 1:
                if filter_pi[i]:
                    acc_poi += 1
                else:
                    miss_poi += 1
            else:
                if filter_pi[i]:
                    miss_clean += 1
                else:
                    acc_clean += 1
        print(f"{name}: #P_rm={acc_poi/(acc_poi+miss_poi)}, # acc_poi={acc_poi}, # miss_poi={miss_poi}, # acc_clean={acc_clean}, # miss_clean={miss_clean}")
        logging.info(
            f"{name}: #P_rm={acc_poi/(acc_poi+miss_poi)}, # acc_poi={acc_poi}, # miss_poi={miss_poi}, # acc_clean={acc_clean}, # miss_clean={miss_clean}")


def main():
    # tsne_visualization_from_file('test', '../record/bad_net_0_1/tsne_visual', 10)
    # tsne_visualization_from_file('train', '../record/bad_net_0_1/tsne_visual', 10)
    save_path = '../record/lc/tsne_visual'
    prefixes = ['train']
    for prefix in prefixes:
        output_array = np.load(os.path.join(save_path, f'{prefix}-tsne_array.npy')).astype(np.float64)
        target_array = np.load(os.path.join(save_path, f'{prefix}-target_array.npy'))
        scatter_visualization(prefix, save_path, 10, output_array, target_array)


if __name__ == '__main__':
    main()
