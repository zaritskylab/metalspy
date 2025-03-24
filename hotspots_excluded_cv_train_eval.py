import numpy as np
import pandas as pd
import warnings
import argparse
import os
import yaml
from sklearn.metrics import roc_auc_score
from data_loading import get_cores
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from histogram_representation import min_max_exclude_hotspots, get_extreme_cores
warnings.filterwarnings("ignore", category=FutureWarning)

cores_dataset = None

class CV_Pipeline:

    def __init__(self, experiments_dir, data_root_directory, seeds) -> None:
        self.experiments_dir = experiments_dir
        self.data_root_directory = data_root_directory
        self.seeds = seeds
    
    def test_specific_subset(self, histogram_size, exclude_outlier_cores, p, predictive_channel, path_dir):
        percentiles = {
            'magnesium': p,
            'iron': p,
            'copper': p,
            'zinc': p,
        }
        if exclude_outlier_cores:
            outlier_cores = get_extreme_cores(self.data_root_directory)
        else:
            outlier_cores = ['Leap095a']
        
        kf = LeaveOneGroupOut()
        core_ids = pd.Series([core.id + core.chunk_id for core in cores_dataset])
        batches = pd.Series([core.batch_date for core in cores_dataset])
        analysis_ids = pd.Series([core.analysis_id for core in cores_dataset])
        y = pd.Series([core.is_responded for core in cores_dataset]) * 1

        filtered_non_extreme_zinc_cores = (~core_ids.isin(outlier_cores)).to_numpy()
        print('filtered_non_extreme_zinc_cores', filtered_non_extreme_zinc_cores.sum())

        cores_dataset_filtered = [core for core, include in zip(cores_dataset, filtered_non_extreme_zinc_cores) if include]
        core_ids = core_ids[filtered_non_extreme_zinc_cores].to_numpy()
        batches = batches[filtered_non_extreme_zinc_cores].to_numpy()
        analysis_ids = analysis_ids[filtered_non_extreme_zinc_cores].to_numpy()
        y = y[filtered_non_extreme_zinc_cores].to_numpy()
        
        is_printed = False
        reports = dict()
        processing = min_max_exclude_hotspots(percentiles, histogram_size, exclude_outlier_cores, self.data_root_directory)
        if not os.path.isdir(os.path.join(path_dir, predictive_channel)):
            os.mkdir(os.path.join(path_dir, predictive_channel))
        
        for i, (train_index, test_index) in enumerate(tqdm(list(kf.split(np.arange(len(core_ids)), y, analysis_ids)))):
            cores_trainset = [cores_dataset_filtered[idx] for idx in train_index]
            cores_testset = [cores_dataset_filtered[idx] for idx in test_index]
            hist_data_train = processing.from_trainset(cores_trainset)
            X_train = hist_data_train[[predictive_channel]].to_numpy()
            X_train = np.stack([np.concatenate(x_i) for x_i in X_train])
            hist_data_test = processing.from_testset(cores_testset)
            X_test = hist_data_test[[predictive_channel]].to_numpy()
            X_test = np.stack([np.concatenate(x_i) for x_i in X_test])
            if not is_printed:
                print('X', X_train.shape)
                print('path_dir', path_dir)
                is_printed = True    
            leap_id = core_ids[test_index]
            y_train = y[train_index]
            for model_seed in self.seeds:
                base_estimator = DecisionTreeClassifier(max_depth=3, min_samples_split=8, class_weight='balanced', random_state=model_seed)
                clf = AdaBoostClassifier(estimator=base_estimator, random_state=model_seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)
                cv_report = pd.DataFrame()
                cv_report['leap-id'] = leap_id
                cv_report['ground-truth'] = y[test_index]
                cv_report['predictions'] = y_pred
                cv_report['prediction-proba-0'] = y_proba[:, 0]
                cv_report['prediction-proba-1'] = y_proba[:, 1]

                if model_seed in reports:
                    reports[model_seed] = pd.concat([reports[model_seed], cv_report]) 
                else:
                    reports[model_seed] = cv_report 
        for model_seed in reports.keys():
            report = reports[model_seed]
            report.to_csv(os.path.join(path_dir, predictive_channel, f'predictions-seed-{model_seed}.csv'))
            table = report.copy(deep=True)
            table['leap-id'] = table['leap-id'].apply(lambda x: x[:-1])
            leap_probabilities = table.groupby('leap-id').aggregate({
                'prediction-proba-0': 'mean',
                'prediction-proba-1': 'mean',
                'ground-truth': 'first',
            }).reset_index()
            results_report = pd.DataFrame({
                'aggregate-all-models-roc-auc-sample': [roc_auc_score(report['ground-truth'].astype('int32'), report['prediction-proba-1'])],
                'aggregate-all-models-roc-auc-patient': [roc_auc_score(leap_probabilities['ground-truth'].astype('int32'), leap_probabilities['prediction-proba-1'])],
            })
            results_csv_path = os.path.join(path_dir, predictive_channel, f'results-seed-{model_seed}.csv')
            print(f'Report dir: {results_csv_path}')
            results_report.to_csv(results_csv_path)

    def experiment(self, config):
        experiment_dir = os.path.join(self.experiments_dir, f'histogram-size-{config["hist_size"]}', f'percentile-{config["p"]}')
        experiment_dir = os.path.join(experiment_dir, 'exclude-outlier-cores' if config['exclude_outlier_cores'] else 'all-cores')
        print('initializing experiment dir', experiment_dir)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        self.test_specific_subset(config["hist_size"], config['exclude_outlier_cores'], config["p"], config["metal"], experiment_dir)

def main():
    global cores_dataset
    default_config_path = './config/default_config.yml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--hist-size', type=str, default=None)
    parser.add_argument('--exclude_outlier_cores', type=str, default=None)
    parser.add_argument('--metal', type=str, default=None)
    parser.add_argument('--p', type=str, default=None)
    
    args = parser.parse_args()
    
    with open(default_config_path, "r") as file:
        config = yaml.safe_load(file)
    
    if args.config is not None:
        with open(default_config_path, "r") as file:
            custom_config = yaml.safe_load(file)
            config['data_root_directory'] = custom_config.get('data_root_directory', config['data_root_directory'])
            config['experiments_root_dir_results'] = custom_config.get('experiments_root_dir_results', config['experiments_root_dir_results'])
            config['hist_size'] = custom_config.get('hist_size', config['hist_size'])
            config['exclude_outlier_cores'] = custom_config.get('exclude_outlier_cores', config['exclude_outlier_cores'])
            config['metal'] = custom_config.get('metal', config['metal'])
            config['p'] = custom_config.get('p', config['p'])
            config['seeds'] = custom_config.get('seeds', config['seeds'])
    
    if args.hist_size is not None:
        config['hist_size'] = int(args.hist_size)
    if args.exclude_outlier_cores is not None:
        config['exclude_outlier_cores'] = args.exclude_outlier_cores == 'True'
    if args.metal is not None:
        config['metal'] = args.metal    
    if args.p is not None:
        config['p'] = float(args.p)

    print('====Config===')
    for key, value in config.items():
        print(f'key:', value, type(value))
    print('=============')
    cores_dataset = get_cores(config["data_root_directory"])
    experiments_dir = config["experiments_root_dir_results"]
    experiments_dir = os.path.join(experiments_dir, 'hotspots_excluded_cv_train_eval')
    print('Saving data to: ', experiments_dir)
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    test = CV_Pipeline(experiments_dir, config["data_root_directory"], config["seeds"])
    test.experiment(config)
main()
