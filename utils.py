import logging
import os
import openpyxl

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def create_logger(experiment_id: str) -> logging.Logger:
    """ 
    Set up a logger for the current experiment.
    """
    # set up directory for the current experiment
    experiment_dir = os.path.join("out", experiment_id)
    log_dir = os.path.join(experiment_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # define filename for log file
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_fn = os.path.join(log_dir, f"{time_str}.log")
    
    # set up logger
    logging.basicConfig(filename=str(log_fn), format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # only add a stream handler if there isn't already one
    if len(logger.handlers) == 1: # <-- file handler is the existing handler
        console = logging.StreamHandler()
        logger.addHandler(console)

    return logger

def norm_calc(tens=torch.tensor, type='euclidian', div=int):

    if type == 'manhattan':
        p=1
    elif type == 'euclidian':
        p=2

    dist_mat = torch.cdist(tens, tens, p=p)    
    up_mat = torch.triu(dist_mat, diagonal=1)
    
    return torch.sum(up_mat) / div

def find_overlap(l1:list, l2:list):
    overlap = 0
    cursor = 0
    for i in range(len(l1)):
        while cursor < len(l2):
            if l1[i] == l2[cursor]:
                overlap += 1
                cursor += 1
                break
            if l1[i] < l2[cursor]:
                break
            cursor += 1

    return overlap


def remove_int(s):
    return s.rstrip('0123456789')


def visualize(table, models, save=False, pre_data='cifar'):
    
    print('\nFirst 5 logged epochs:\n', table.T.head())
    print('\nLast 5 logged epochs:\n', table.T.tail())

    for model in models:
        data = table.loc[model]
        x = data.index

        if pre_data == 'noise':
            # Separate bias_%, down_acc, pair_e_dist
            data_np = np.vstack(data)
            bias, down, pair_dist = data_np[:, 0], data_np[:, 1], data_np[:, 2]

            # Save each independently
            pd.DataFrame(bias).to_excel(os.path.join('scores', f'{model}'+'__bias_scores.xlsx'))
            pd.DataFrame(down).to_excel(os.path.join('scores', f'{model}'+'__down_scores.xlsx'))
            pd.DataFrame(pair_dist).to_excel(os.path.join('scores', f'{model}'+'__pair_dist_scores.xlsx'))

            # Plot evolutions
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
            fig.suptitle('Evolution of scores for {} throughout training'.format(model), fontsize=18)

            ax1.plot(x, bias)
            ax1.set_title('Shape bias')
            ax1.set_ylabel('Bias percentage (%)')
            ax1.set_xlabel('Epochs')

            ax2.plot(x, down)
            ax2.set_title('Downstream test accuracy')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_xlabel('Epochs')

            ax3.plot(x, pair_dist)
            ax3.set_title('Distance between augmented downstream embedding pairs')
            ax3.set_ylabel('Distance')
            ax3.set_xlabel('Epochs')

            fig.tight_layout()
            # subplots_adjust(hspace=1.0, wspace=1.0)
            if save: plt.savefig(os.path.join('scores', 'Results_{}.png'.format(model)), dpi=300)
            plt.show()


        else:
            # Separate pre_acc, bias_%, down_acc, pair_e_dist
            data_np = np.vstack(data)
            pre, bias, down, pair_dist = data_np[:, 0], data_np[:, 1], data_np[:, 2], data_np[:, 3]

            # Save each independently
            pd.DataFrame(pre).to_excel(os.path.join('scores', f'{model}'+'__pre_scores.xlsx'))
            pd.DataFrame(bias).to_excel(os.path.join('scores', f'{model}'+'__bias_scores.xlsx'))
            pd.DataFrame(down).to_excel(os.path.join('scores', f'{model}'+'__down_scores.xlsx'))
            pd.DataFrame(pair_dist).to_excel(os.path.join('scores', f'{model}'+'__pair_dist_scores.xlsx'))

            # Plot evolutions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
            fig.suptitle('Evolution of scores for {} throughout training'.format(model), fontsize=18)
            
            ax1.plot(x, pre)
            ax1.set_title('Pretext test accuracy')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_xlabel('Epochs')

            ax2.plot(x, bias)
            ax2.set_title('Shape bias')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_xlabel('Epochs')


            fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14,6))
            fig2.suptitle('Evolution of scores for {} throughout training'.format(model), fontsize=18)
            
            ax3.plot(x, down)
            ax3.set_title('Downstream test accuracy')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_xlabel('Epochs')

            ax4.plot(x, pair_dist)
            ax4.set_title('Distance between augmented downstream embedding pairs')
            ax4.set_ylabel('Distance')
            ax4.set_xlabel('Epochs')

            fig.tight_layout()
            fig2.tight_layout()
            if save: 
                fig.savefig(os.path.join('scores', 'Results_1_{}.png'.format(model)), dpi=300)
                fig2.savefig(os.path.join('scores', 'Results_2_{}.png'.format(model)), dpi=300)
            plt.show()

