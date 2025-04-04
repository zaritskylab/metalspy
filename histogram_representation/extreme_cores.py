import pandas as pd
import numpy as np
import os

def get_extreme_cores(root_dir):
    df = pd.read_csv(os.path.join(root_dir, 'cores-with-outlier-distribution-tissue-median.csv'))
    extreme_cores_per_metal = {
        'magnesium': df.loc[~df['magnesium sample 95 percentile and first bin removed'].isna(), 'leap-id'].to_list(),
        'iron': df.loc[~df['iron sample 95 percentile and first bin removed'].isna(), 'leap-id'].to_list(),
        'copper': df.loc[~df['copper sample 95 percentile and first bin removed'].isna(), 'leap-id'].to_list(),
        'zinc': df.loc[~df['zinc sample 95 percentile and first bin removed'].isna(), 'leap-id'].to_list(),
    }
    return np.unique([leap_id for leap_ids in extreme_cores_per_metal.values() for leap_id in leap_ids]).tolist()

def get_cores(exclude_outlier_cores, root_dir):
    if exclude_outlier_cores:
        outlier_cores = get_extreme_cores(root_dir)
    else:
        outlier_cores = ['Leap095a']
    return outlier_cores