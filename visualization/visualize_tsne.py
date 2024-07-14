import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from openTSNE import TSNE as op_TSNE


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--save_path", type=str)
    return parser


def scatter_visualization(prefix, save_path, num_class, output_array, target_array, dof=1):
    plt.rcParams['figure.figsize'] = 10, 10
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure()
    point_list = [[] for i in range(num_class + 1)]
    colors = ["#f94144", "#f3722c", "#f8961e", "#f9844a", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590",
              "#277da1"]
    for i in range(len(target_array[:, 0])):
        t = target_array[i, 0]
        point = output_array[i, :]
        point_list[t].append(point)
    poison_rep = np.array(point_list[num_class])
    if len(poison_rep) > 500:
        plt.scatter(poison_rep[:, 0], poison_rep[:, 1], c='#161a1d', marker='x', label='poisoned', alpha=0.1,
                    edgecolors='none')
    visual_num = 0
    for i in range(num_class):
        if len(point_list[i]) > 0:
            visual_num += 1
    step = len(colors) // visual_num
    new_colors = []
    for i in range(visual_num):
        new_colors.append(colors[i * step])
    colors = new_colors
    j = 0
    for i in range(num_class):
        if len(point_list[i]) > 0:
            representation = np.array(point_list[i])
            plt.scatter(representation[:, 0], representation[:, 1], c=colors[j], label=f'Label{i}', alpha=0.4,
                        edgecolors='none')
            j += 1
    if len(poison_rep) <= 500:
        plt.scatter(poison_rep[:, 0], poison_rep[:, 1], c='#161a1d', marker='x', label='poisoned', alpha=0.5,
                    edgecolors='none')
    plt.legend(fontsize=25, markerscale=2, loc='lower right')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig(os.path.join(save_path, prefix + f'-tsne_dof-{dof}.pdf'), bbox_inches='tight')


def tsne_visualization(model, p_only, cl_only, num_class, prefix, device, save_path):
    model.to(device)
    model.eval()
    out_target = []
    out_output = []
    tsne_path = os.path.join(save_path, 'tsne_visual')
    if not os.path.exists(tsne_path):
        os.mkdir(tsne_path)
    if not os.path.exists(os.path.join(tsne_path, prefix + '-out_array.npy')) \
            or not os.path.exists(os.path.join(tsne_path, prefix + '-target_array.npy')):
        for batch_idx, (inputs, targets, *additional_info) in enumerate(cl_only):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            penultimate_outputs = model.get_penultimate()
            output_np = penultimate_outputs.data.cpu().numpy()
            target_np = targets.data.cpu().numpy()
            out_output.append(output_np)
            out_target.append(target_np[:, np.newaxis])

        for batch_idx, (inputs, targets, *additional_info) in enumerate(p_only):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            penultimate_outputs = model.get_penultimate()
            output_np = penultimate_outputs.data.cpu().numpy()
            target_ = targets.data.cpu().numpy()
            target_np = np.zeros_like(target_)
            target_np = target_np + num_class
            out_output.append(output_np)
            out_target.append(target_np[:, np.newaxis])

        output_array = np.concatenate(out_output, axis=0)
        target_array = np.concatenate(out_target, axis=0)

        np.save(os.path.join(tsne_path, prefix + '-out_array.npy'), output_array, allow_pickle=False)
        np.save(os.path.join(tsne_path, prefix + '-target_array.npy'), target_array, allow_pickle=False)
    # feature = np.load('./label_smooth1.npy').astype(np.float64)
    # target = np.load('./label_smooth_target1.npy')
    if num_class <= 10:
        # if not os.path.exists(os.path.join(tsne_path, 'train-tsne_dof-1.pdf')):
        #     tsne_visualization_from_file(prefix, tsne_path, num_class)
        # if not os.path.exists(os.path.join(tsne_path, 'train-visual_num3-tsne_dof-1.pdf')):
        # if not os.path.exists(os.path.join(tsne_path, 'train-visual_num5-tsne_dof-1.pdf')):
        # tsne_visualization_ByClassNum_from_file(prefix, tsne_path, num_class, 5)
        tsne_visualization_ByClassNum_from_file(prefix, tsne_path, num_class, 3)
        # tsne_visualization_ByClassNum_from_file(prefix, tsne_path, num_class, 1)
    elif num_class == 100:
        if not os.path.exists(os.path.join(tsne_path, 'train-visual_num5-tsne_dof-1.pdf')):
            tsne_visualization_ByClassNum_from_file(prefix, tsne_path, num_class, 5)
            tsne_visualization_ByClassNum_from_file(prefix, tsne_path, num_class, 3)
        if not os.path.exists(os.path.join(tsne_path, 'train-visual_num10-tsne_dof-1.pdf')):
            tsne_visualization_ByClassNum_from_file(prefix, tsne_path, num_class, 10)
        tsne_visualization_ByClassNum_from_file(prefix, tsne_path, num_class, 1)
    else:
        tsne_visualization_TargetByClassNum_from_file(prefix, tsne_path, num_class, 4)
    # open_tsne_visualization_from_file(prefix, tsne_path, num_class, dof=5)
    # open_tsne_visualization_from_file(prefix, tsne_path, num_class, dof=100)


def tsne_visualization_ByClassNum_from_file(prefix, save_path, num_class, visual_num):
    output_array = np.load(os.path.join(save_path, f'{prefix}-out_array.npy')).astype(np.float64)
    target_array = np.load(os.path.join(save_path, f'{prefix}-target_array.npy'))
    print('test_Pred shape :', output_array.shape)
    print('test_Target shape :', target_array.shape)
    output_array_visual, target_array_visual = [], []
    for i in range(len(target_array)):
        # if (0 < target_array[i] <= 5) or target_array[i] == num_class:
        if target_array[i] < visual_num or target_array[i] == num_class:
            output_array_visual.append(output_array[i])
            target_array_visual.append([int(target_array[i])])

    target_array_visual = np.array(target_array_visual)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    output_array_visual = tsne.fit_transform(np.array(output_array_visual))
    # np.save(os.path.join(save_path, prefix + '-tsne_array.npy'), output_array, allow_pickle=False)
    scatter_visualization(prefix + f"-visual_num{visual_num}",
                          save_path, num_class, output_array_visual, target_array_visual)


def tsne_visualization_TargetByClassNum_from_file(prefix, save_path, num_class, target):
    output_array = np.load(os.path.join(save_path, f'{prefix}-out_array.npy')).astype(np.float64)
    target_array = np.load(os.path.join(save_path, f'{prefix}-target_array.npy'))
    print('test_Pred shape :', output_array.shape)
    print('test_Target shape :', target_array.shape)
    output_array_visual, target_array_visual = [], []
    for i in range(len(target_array)):
        if (target_array[i] == target) or target_array[i] == num_class:
            output_array_visual.append(output_array[i])
            target_array_visual.append([int(target_array[i])])

    target_array_visual = np.array(target_array_visual)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    output_array_visual = tsne.fit_transform(np.array(output_array_visual))
    # np.save(os.path.join(save_path, prefix + '-tsne_array.npy'), output_array, allow_pickle=False)
    scatter_visualization(prefix + f"-visual_num1",
                          save_path, num_class, output_array_visual, target_array_visual)


def tsne_visualization_from_file(prefix, save_path, num_class):
    output_array = np.load(os.path.join(save_path, f'{prefix}-out_array.npy')).astype(np.float64)
    target_array = np.load(os.path.join(save_path, f'{prefix}-target_array.npy'))
    print('test_Pred shape :', output_array.shape)
    print('test_Target shape :', target_array.shape)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    output_array = tsne.fit_transform(output_array)
    np.save(os.path.join(save_path, prefix + '-tsne_array.npy'), output_array, allow_pickle=False)
    scatter_visualization(prefix, save_path, num_class, output_array, target_array)


def open_tsne_visualization_from_file(prefix, save_path, num_class, dof=1):
    output_array = np.load(os.path.join(save_path, f'{prefix}-out_array.npy')).astype(np.float64)
    target_array = np.load(os.path.join(save_path, f'{prefix}-target_array.npy'))
    print('test_Pred shape :', output_array.shape)
    print('test_Target shape :', target_array.shape)

    tsne = op_TSNE(n_components=2, random_state=0, dof=dof)
    output_array = tsne.fit(output_array)
    np.save(os.path.join(save_path, prefix + f'-open_tsne_array-dof_{dof}.npy'), output_array, allow_pickle=False)
    scatter_visualization(prefix + "-open", save_path, num_class, output_array, target_array, dof=dof)

