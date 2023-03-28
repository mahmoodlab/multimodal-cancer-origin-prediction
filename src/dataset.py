import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SequentialSampler


class MultiModalDataset(Dataset):
    def __init__(self, gt_csv, split_csv, label_dict, histology_feature_path, genomic_feature_path, split_name='train', site_name=None, balance_met=False, gender=True):
        self.label_dict = label_dict
        self.balance_met = balance_met
        self.histology_feature_path = histology_feature_path
        self.genomic_feature_path = genomic_feature_path
        self.split_name = split_name
        self.site_name = site_name
        self.balance_met = balance_met
        gt_df = pd.read_csv(gt_csv)

        if (split_csv is None) or (split_csv == 'None'):
            case_list = gt_df['case_id'].to_list()
        else:
            split_df = pd.read_csv(split_csv)
            split_df.columns = ['case_id', 'train', 'valid', 'test']
            if split_name == 'all':
                case_list = split_df[split_df['train']]['case_id'].to_list()
                case_list.extend(split_df[split_df['valid']]['case_id'].to_list())
                case_list.extend(split_df[split_df['test']]['case_id'].to_list())
            else:
                split_df = split_df[split_df[split_name]]
                case_list = split_df['case_id'].to_list()

        g_features = pd.read_csv(self.genomic_feature_path)
        if gender:
            columns = [col for col in g_features.columns if col not in ['case_id']]
        else:
            columns = [col for col in g_features.columns if col not in ['case_id', 'sex']]
        genomic_features = {}
        for i, row in g_features.iterrows():
            genomic_features[row['case_id']] = np.array(row[columns]).astype(np.float32)

        h_features = pd.read_csv(self.histology_feature_path)
        if gender:
            columns = [col for col in h_features.columns if col not in ['case_id']]
        else:
            columns = [col for col in h_features.columns if col not in ['case_id', 'sex']]
        histology_features = {}
        for i, row in h_features.iterrows():
            histology_features[row['case_id']] = np.array(row[columns]).astype(np.float32)

        self.default_g_feature = np.zeros((len(columns))).astype(np.float32)
        self.default_h_feature = np.zeros((h_features.shape[1])).astype(np.float32)

        self.case_level_dataset(gt_df, case_list, genomic_features, histology_features)

    def case_level_dataset(self, gt_df, case_list, genomic_features, histology_features):
        self.x = []
        self.labels = []
        self.site_labels = []
        self.genomic_features = {}
        self.histology_features = {}
        for i, row in gt_df.iterrows():
            if (row['case_id'] in case_list):
                    # and (row['case_id'] in genomic_features.keys())
                    # and (row['case_id'] in histology_features.keys())):
                if row['case_id'] not in self.x:
                    flag = True
                    if not (self.site_name is None or self.site_name != 'None'):
                        flag = False
                        if row['site'].find(self.site_name) != -1:
                            flag = True
                    if flag:
                        self.x.append(row['case_id'])
                        self.labels.append(self.label_dict[row['label']])
                        if row['site'].find('Metastatic') != -1:
                            self.site_labels.append(1)
                        else:
                            self.site_labels.append(0)
                        self.genomic_features[row['case_id']] = genomic_features[row['case_id']]
                        self.histology_features[row['case_id']] = histology_features[row['case_id']]

    def stats(self, split_name='test', site_name=None):
        row_labels = []
        class_counts = []
        for label in self.label_dict.keys():
            row_labels.append(label)
            class_counts.append(self.labels.count(self.label_dict[label]))
        if site_name is not None:
            col_name = '%s_%s' % (split_name, site_name)
        else:
            col_name = split_name
        return pd.DataFrame(class_counts, index=row_labels, columns=[col_name])

    @staticmethod
    def get_data_loader(dataset, batch_size=4, training=False):
        if training:
            n = float(len(dataset))
            weight = [0] * int(n)
            if dataset.balance_met:
                weight_per_class = []
                for c in range(len(dataset.label_dict)):
                    if dataset.labels.count(c) > 0:
                        weight_per_class.append(n / dataset.labels.count(c))
                    else:
                        weight_per_class.append(0)
                for idx in range(len(dataset)):
                    label = dataset.site_labels[idx]
                    weight[idx] = weight_per_class[label]
            else:
                weight_per_class = []
                for c in range(len(dataset.label_dict)):
                    if dataset.labels.count(c) > 0:
                        weight_per_class.append(n / dataset.labels.count(c))
                    else:
                        weight_per_class.append(0)
                for idx in range(len(dataset)):
                    label = dataset.labels[idx]
                    weight[idx] = weight_per_class[label]

            weight = torch.DoubleTensor(weight)
            loader = DataLoader(dataset, batch_size=batch_size, sampler=WeightedRandomSampler(weight, len(weight)), drop_last=True, num_workers=4)
        else:
            loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), drop_last=False, num_workers=4)

        return loader

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        case_id, label = self.x[idx], self.labels[idx]
        h_flag, g_flag = False, False

        # genomic features
        if case_id in self.genomic_features.keys():
            g_feature = torch.from_numpy(self.genomic_features[case_id])
            g_flag = True
        else:
            g_feature = torch.from_numpy(self.default_g_feature)

        # histology features
        if case_id in self.histology_features.keys():
            h_features = torch.from_numpy(self.histology_features[case_id])
            h_flag = True
        else:
            h_features = torch.from_numpy(self.default_h_feature)

        return g_feature, h_features, label, case_id, g_flag, h_flag
