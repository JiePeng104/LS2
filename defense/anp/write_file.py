import argparse

import torch
import os


def get_args():
    # set the basic parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=str, help='cuda, cpu')
    arg = parser.parse_args()
    print(arg)
    return arg


args = get_args()
model_path = os.getcwd() + f'/record/{args.record}/anp/'
model_result = os.listdir(model_path)
model_result.sort()
with open(model_path+'summary.txt', 'w') as f:
    for model in model_result:
        if model == 'summary.txt':
            continue
        result = torch.load(model_path+model)
        f.write("*******  %s  *******\n" % model)
        f.write("{asr: %f \n" % result['asr'])
        f.write(" acc: %f \n" % result['acc'])
        f.write(" ra: %f } \n\n" % result['ra'])
