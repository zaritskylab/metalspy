import pandas as pd
import numpy as np
from typing import List
from .data import Leap
from .extreme_cores import get_cores
from .min_max_intersection import get_core_pixel_mask
from sklearn.preprocessing import MinMaxScaler

class min_max_shift_5_7_iqr_intersection:
    def __init__(self, metal_to_percentile: dict, histogram_size=None, exclude_outlier_cores=True):
        self.histogram_size = histogram_size
        self.exclude_outlier_cores = exclude_outlier_cores
        self.metal_to_percentile = metal_to_percentile
        self.channel_properties = None
    
    def from_trainset(self, cores: List[Leap]):
        table, channel_properties = self.get_histogram(cores)
        self.channel_properties = channel_properties
        return table
    
    def from_testset(self, cores: List[Leap]):
        table, _ = self.get_histogram(cores, self.channel_properties)
        return table

    def get_histogram(self, cores: List[Leap], channel_properties=None):
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
        if channel_properties is None:
            channel_properties = dict()
            for i, channel in enumerate(['magnesium', 'iron', 'copper', 'zinc']):
                all_pixels = []
                outlier_cores_of_metal = outlier_cores
                for core in cores:
                    pixels = core.pixels[channel]
                    pixel_mask = get_core_pixel_mask(core, self.metal_to_percentile)
                    core_df = pixels[pixel_mask]
                    if (core.id + core.chunk_id) not in outlier_cores_of_metal:
                        all_pixels.append(core_df)
                medians = np.array([core_df.median() for core_df in all_pixels])
                medians_scaler = MinMaxScaler()
                medians_scaler.fit(medians.reshape(medians.shape[0], 1))
                medians = medians_scaler.transform(medians.reshape(medians.shape[0], 1))
                mean_of_medians = np.mean(medians)
                p75 = np.array([core_df.quantile(q=0.75) for core_df in all_pixels])
                p75_scaler = MinMaxScaler()
                p75_scaler.fit(p75.reshape(p75.shape[0], 1))
                p75 = p75_scaler.transform(p75.reshape(p75.shape[0], 1))
                mean_of_percentile_75 = np.mean(p75)
                iqr = np.array([core_df.quantile(q=0.75) - core_df.quantile(q=0.25) for core_df in all_pixels])
                iqr_scaler = MinMaxScaler()
                iqr_scaler.fit(iqr.reshape(iqr.shape[0], 1))
                iqr = iqr_scaler.transform(iqr.reshape(iqr.shape[0], 1))
                mean_of_iqr = np.mean(iqr)
                channel_properties[channel] = {
                    'mean-of-medians': mean_of_medians,
                    'medians-scaler': medians_scaler,
                    'mean-of-percentile-75': mean_of_percentile_75,
                    'percentile-75-scaler': p75_scaler,
                    'mean-of-iqr': mean_of_iqr,
                    'iqr-scaler': iqr_scaler,
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
                    all_data_bin_edges = np.histogram_bin_edges(core_df, bins=bins_amount)
                    pixel_counts = np.histogram(core_df, bins=all_data_bin_edges)[0]
                    pixel_sum = pixel_counts.sum()
                    hist = pixel_counts / pixel_sum

                    median = core_df.median()
                    scaler = channel_properties[channel]['medians-scaler']
                    median = scaler.transform([[median]])[0, 0]
                    shift_median = median - channel_properties[channel]['mean-of-medians']
                    
                    percentile_75 = core_df.quantile(q=0.75)
                    percentile_25 = core_df.quantile(q=0.25)
                    iqr = percentile_75 - percentile_25
                    
                    scaler = channel_properties[channel]['percentile-75-scaler']
                    percentile_75 = scaler.transform([[percentile_75]])[0, 0]
                    
                    scaler = channel_properties[channel]['iqr-scaler']
                    iqr = scaler.transform([[iqr]])[0, 0]
                    
                    shift_percentile_75 = percentile_75 - channel_properties[channel]['mean-of-percentile-75']
                    
                    shift_iqr = iqr - channel_properties[channel]['mean-of-iqr']
                    
                    hist = np.concatenate([[shift_median, shift_percentile_75, shift_iqr], hist])
                    table[channel].append(hist)
        table = pd.DataFrame(table)
        table = table.sort_values(by=['is-responded'])
        table = table.sort_values(by='leap-id').reset_index(drop=True)
        table['is-responded'] = table['is-responded'] * 1
        return table, channel_properties