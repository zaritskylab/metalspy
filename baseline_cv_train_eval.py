import numpy as np
import pandas as pd
import warnings
import argparse
import os
from sklearn.metrics import roc_auc_score
from data_loading import get_cores
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from histogram_representation import min_max_intersection, get_extreme_cores
warnings.filterwarnings("ignore", category=FutureWarning)

cores_dataset = get_cores()

class CV_Pipeline:

    def __init__(self, experiments_dir) -> None:
        self.experiments_dir = experiments_dir

    def test_specific_subset(self, histogram_size, exclude_outlier_cores, p, predictive_channel_job, path_dir):
        percentiles = {
            'magnesium': p,
            'iron': p,
            'copper': p,
            'zinc': p,
        }
        if exclude_outlier_cores:
            outlier_cores = get_extreme_cores()
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
        
        for predictive_channel in [predictive_channel_job]:
            reports = dict()
            processing = min_max_intersection(percentiles, histogram_size, exclude_outlier_cores)
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
                leap_id = core_ids[test_index]
                y_train = y[train_index]
                for model_seed in [11, 19, 21, 22, 31, 36, 38, 54, 67, 82, 85, 88, 92, 94, 96, 112, 116, 140, 148, 156, 161, 177, 178, 212, 223, 225, 240, 242, 249, 276, 289, 293, 294, 300, 306, 309, 311, 320, 338, 342, 349, 358, 372, 373, 374, 382, 395, 418, 440, 444, 445, 465, 479, 480, 492, 494, 526, 567, 569, 586, 596, 600, 602, 606, 620, 633, 637, 641, 645, 647, 664, 689, 697, 708, 748, 753, 777, 794, 804, 812, 828, 831, 843, 858, 861, 875, 889, 891, 894, 897, 904, 917, 947, 962, 982, 983, 985, 987, 990, 998]:
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

    def experiment(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hist-size', type=str, required=True)
        parser.add_argument('--exclude_outlier_cores', type=str, required=True)
        parser.add_argument('--metal', type=str, required=True)
        parser.add_argument('--p', type=str, required=True)

        args = parser.parse_args()
        histogram_size = int(args.hist_size)
        print('args.hist_size value is', args.hist_size, histogram_size)
        print('args.exclude_outlier_cores value is ', args.exclude_outlier_cores)
        exclude_outlier_cores = args.exclude_outlier_cores == 'True'
        print('args.exclude_outlier_cores', exclude_outlier_cores, type(exclude_outlier_cores))

        metal = args.metal
        p = float(args.p)

        print('exclude_outlier_cores', exclude_outlier_cores, 'histogram_size', histogram_size)
        experiment_dir = os.path.join(self.experiments_dir, f'histogram-size-{histogram_size}', f'p-{p}')
        experiment_dir = os.path.join(experiment_dir, 'exclude-outlier-cores' if exclude_outlier_cores else 'all-cores')
        print('initializing experiment dir', experiment_dir)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        self.test_specific_subset(histogram_size, exclude_outlier_cores, p, metal, experiment_dir)

def main():
    experiments_dir = os.path.join('.', 'results', 'baseline_cv_train_eval')
    print('Saving data to: ', experiments_dir)
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)
    test = CV_Pipeline(experiments_dir)
    test.experiment()
main()
