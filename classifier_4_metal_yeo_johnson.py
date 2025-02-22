import pandas as pd
import numpy as np
import warnings
import os
import argparse
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)

def simple_classifiers(metal_to_hist_size, path):
    stats = {
        'probabilities-average': [],
    }
    for model_seed in tqdm([11, 19, 21, 22, 31, 36, 38, 54, 67, 82, 85, 88, 92, 94, 96, 112, 116, 140, 148, 156, 161, 177, 178, 212, 223, 225, 240, 242, 249, 276, 289, 293, 294, 300, 306, 309, 311, 320, 338, 342, 349, 358, 372, 373, 374, 382, 395, 418, 440, 444, 445, 465, 479, 480, 492, 494, 526, 567, 569, 586, 596, 600, 602, 606, 620, 633, 637, 641, 645, 647, 664, 689, 697, 708, 748, 753, 777, 794, 804, 812, 828, 831, 843, 858, 861, 875, 889, 891, 894, 897, 904, 917, 947, 962, 982, 983, 985, 987, 990, 998]):
        metal = 'magnesium'
        hist_size = metal_to_hist_size[metal]
        df_mg = pd.read_csv(f'{path}\\histogram-size-{hist_size}\\all-cores\\min-max-shift-5-7-iqr-intersection\\{metal}\\predictions-seed-{model_seed}.csv')
        metal = 'iron'
        hist_size = metal_to_hist_size[metal]
        df_ir = pd.read_csv(f'{path}\\histogram-size-{hist_size}\\all-cores\\min-max-shift-5-7-iqr-intersection\\{metal}\\predictions-seed-{model_seed}.csv')
        metal = 'copper'
        hist_size = metal_to_hist_size[metal]
        df_cp = pd.read_csv(f'{path}\\histogram-size-{hist_size}\\all-cores\\min-max-shift-5-7-iqr-intersection\\{metal}\\predictions-seed-{model_seed}.csv')
        metal = 'zinc'
        hist_size = metal_to_hist_size[metal]
        df_zn = pd.read_csv(f'{path}\\histogram-size-{hist_size}\\all-cores\\min-max-shift-5-7-iqr-intersection\\{metal}\\predictions-seed-{model_seed}.csv')
        
        y_true = df_mg['ground-truth']
        
        y_proba = np.stack([
            df_mg['prediction-proba-1'],
            df_ir['prediction-proba-1'],
            df_cp['prediction-proba-1'],
            df_zn['prediction-proba-1'],
        ])
        y_proba = np.average(y_proba, axis=0)
        
        auc = roc_auc_score(y_true, y_proba)
        stats['probabilities-average'].append(auc)
        
    return pd.DataFrame(stats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root-path', type=str, required=True)
    args = parser.parse_args()
    
    experiment_dir = args.experiment_root_path
    metal_to_hist_size = {
        'magnesium': 20,
        'iron': 20,
        'copper': 20,
        'zinc': 20,
    }
    
    output_file = './yeo_johnson.csv'
    output_file = os.path.join('.', 'results', '4-metals', output_file)
    results_df = simple_classifiers(metal_to_hist_size, experiment_dir)
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()