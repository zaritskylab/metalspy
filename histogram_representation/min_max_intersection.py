import pandas as pd
import numpy as np
from typing import List
from .data import Leap
from .extreme_cores import get_cores

def get_core_pixel_mask(core: Leap, metal_to_percentile: dict):
    pixel_mask_upper_outliers = pd.Series([True] * len(core.pixels.index))
    pixel_mask_lower_outliers = pd.Series([True] * len(core.pixels.index))
    for i, channel in enumerate(['magnesium', 'iron', 'copper', 'zinc']):
        pixels = core.pixels[channel]
        percentile = metal_to_percentile[channel]
        edges = np.histogram_bin_edges(pixels.to_numpy(), bins='fd')
        threshold = edges[1]
        pixel_mask_upper_outliers = pixel_mask_upper_outliers & (pixels >= pixels.quantile(q=percentile)) 
        pixel_mask_lower_outliers = pixel_mask_lower_outliers & (pixels <= threshold)
    pixel_mask = (~pixel_mask_upper_outliers) & (~pixel_mask_lower_outliers)
    return pixel_mask

class min_max_intersection:
    def __init__(self, metal_to_percentile: dict, histogram_size=None, exclude_outlier_cores=True):
        self.histogram_size = histogram_size
        self.exclude_outlier_cores = exclude_outlier_cores
        self.metal_to_percentile = metal_to_percentile
    
    def from_trainset(self, cores: List[Leap]):
        table = self.get_histogram(cores)
        return table
    
    def from_testset(self, cores: List[Leap]):
        table = self.get_histogram(cores)
        return table

    def get_histogram(self, cores: List[Leap]):
        outlier_cores = get_cores(self.exclude_outlier_cores)
        bins_amount = self.histogram_size
        table = {
            'leap-id': [],
            'is-responded': [],
            'magnesium': [],
            'iron': [],
            'copper': [],
            'zinc': [],
        }
        for core in cores:
            table['leap-id'].append(core.id + core.chunk_id)
            table['is-responded'].append(core.is_responded)
            pixel_mask = get_core_pixel_mask(core, self.metal_to_percentile)
            for i, channel in enumerate(['magnesium', 'iron', 'copper', 'zinc']):
                core_df = core.pixels.loc[pixel_mask, channel]
                if (core.id + core.chunk_id) in outlier_cores:
                    table[channel].append(None)
                else:
                    # if pixel_mask.sum() / len(core.pixels.index) < 0.25:
                    #     print(core.id + core.chunk_id, f'{pixel_mask.sum() / len(core.pixels.index):.2%}')
                    all_data_bin_edges = np.histogram_bin_edges(core_df, bins=bins_amount)
                    pixel_counts = np.histogram(core_df, bins=all_data_bin_edges)[0]
                    pixel_sum = pixel_counts.sum()
                    hist = pixel_counts / pixel_sum
                    table[channel].append(hist)
        table = pd.DataFrame(table)
        table = table.sort_values(by=['is-responded'])
        table = table.sort_values(by='leap-id').reset_index(drop=True)
        table['is-responded'] = table['is-responded'] * 1
        return table