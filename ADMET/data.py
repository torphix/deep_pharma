import torch
import pandas as pd
from tdc.single_pred import ADME, Tox
from torch.utils.data import DataLoader
from tdc.utils import retrieve_dataset_names
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler
from utils import MolecularVocab, numerate_features, smiles_to_graph


class ADMETDataset(Dataset):
    def __init__(self,
                df:pd.DataFrame,
                dataset_name,
                use_coords,
                atom_features,
                bond_features):
        super().__init__()
        vocab = MolecularVocab()
        df['graph'] = df['Drug'].apply(
                smiles_to_graph, 
                node_vocab=vocab.atom_stoi,
                coords=use_coords,
                node_features=atom_features, 
                bond_features=bond_features)
        # Featurize
        src_df = pd.DataFrame(
            df['graph'].apply(numerate_features).tolist(),
            index=df.index,
            columns=['drug', 'edge_idx'])
        self.drugs = src_df['drug']
        self.edge_idxs = src_df['edge_idx']
        # self.edge_attrs = src_df['edge_attr']
        self.smiles = df['Drug']
        self.tgt = df['Y']
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        data = Data(x=torch.tensor(self.drugs[idx]), 
                    edge_index=torch.tensor(self.edge_idxs[idx]),
                    y=torch.tensor(self.tgt[idx]),
                    dataset_name=self.dataset_name,
                    smiles=self.smiles[idx])
        return data

    @staticmethod
    def collate_fn(batch):
        datasets = {}
        # Init dict
        for data in batch:
            datasets[data.dataset_name] = []
        # Add relevant datapoints
        for data in batch:
            values = datasets[data.dataset_name]
            values.append(data)
            datasets[data.dataset_name] = values
        # Convert to batch
        for k,v in datasets.items():
            datasets[k] = Batch.from_data_list(v)
        return datasets


class DatasetHandler():
    def __init__(
            self, 
            split_size=[1],
            balance_labels=None,
            oversample=None,
            use_coords=False, 
            atom_features = ['atomic_mass', 'implicit_valence', 'explicit_valence', 'formal_charge', 'hybridization', 'is_aromatic', 'is_isotope', 'chiral_tag', 'vocab_idx'],
            bond_features = ['bond_type', 'is_aromatic', 'is_conjugated', 'bond_stereo'],
            seed=42):

        self.adme_names = retrieve_dataset_names('ADME')
        self.tox_names = retrieve_dataset_names('Tox')

        self.split_size = split_size
        self.oversample = oversample
        self.balance_labels = balance_labels
        self.use_coords = use_coords
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.seed = seed

    def load_dataset(self, dataset_name, label_name=None):
        # Fetch
        dataset_name = dataset_name.lower()
        if dataset_name in self.adme_names:
            dataset = ADME(dataset_name, label_name=label_name)
        elif dataset_name in self.tox_names:
            dataset = Tox(dataset_name, label_name=label_name)
        else:
            raise ValueError(f'Dataset: {dataset_name} not found!')
        # Process & split
        if self.balance_labels:
            dataset = dataset.balanced(oversample=self.oversample, seed=self.seed)
        else: dataset = dataset.get_data()
        if int(self.split_size[0]) != 1:
            datasets = self.split_dataset(dataset)
        else:
            datasets = [dataset]
        datasets = [
            ADMETDataset(df, dataset_name, self.use_coords, self.atom_features, self.bond_features) 
            for df in datasets]
        return datasets
    
    def load_multiple_datasets(self, dataset_names):
        train, val, test  = [], [], []
        for name in dataset_names:
            dataset = self.load_dataset(name)
            if len(dataset) == 3:
                train.append(dataset[0]), val.append(dataset[1]), test.append(dataset[2])
            elif len(dataset) == 2:
                train.append(dataset[0]), val.append(dataset[1])
            elif len(dataset) == 1:
                train.append(dataset[0])
        if len(test) != 0: test = ConcatDataset(test)
        if len(val) != 0: val = ConcatDataset(val)
        if len(train) != 0: train = ConcatDataset(train)
        if len(self.split_size) == 3: return train, val, test
        elif len(self.split_size) == 2: return train, val
        elif len(self.split_size) == 1: return train

    def load_dataloaders(self, dataset_names, dataloader_params):
        if len(self.split_size) == 3:
            train, val, test = self.load_multiple_datasets(dataset_names)
            train_dl, val_dl, test_dl = (
                DataLoader(train, **dataloader_params, collate_fn=ADMETDataset.collate_fn),
                DataLoader(val, shuffle=False, batch_size=64, collate_fn=ADMETDataset.collate_fn),
                DataLoader(test, shuffle=False, batch_size=64, collate_fn=ADMETDataset.collate_fn)) 
            return train_dl, val_dl, test_dl
        elif len(self.split_size) == 2:
            train, val = self.load_multiple_datasets(dataset_names)
            train_dl, val_dl = (
                DataLoader(train, **dataloader_params, collate_fn=ADMETDataset.collate_fn),
                DataLoader(val, shuffle=False, batch_size=64, collate_fn=ADMETDataset.collate_fn))
            return train_dl, val_dl
        elif len(self.split_size) == 1:
            train = self.load_multiple_datasets(dataset_names)
            train = self.load_multiple_datasets(dataset_names)
            train_dl = (
                DataLoader(train, **dataloader_params, collate_fn=ADMETDataset.collate_fn))
            return train_dl

    def misclassified_to_dfs(self, misclassified:dict):
        dfs = {}
        for k, samples in misclassified.items():
            df = ADME(k).get_data()
            dfs[k] = df.loc[df['Drug'].isin(samples)].reset_index(drop=True)
        return dfs

    def df_to_dataset(self, 
                    df, name, 
                    use_coords=False,
                    atom_features=[
                        'atomic_mass','implicit_valence',
                        'explicit_valence','formal_charge',
                        'hybridization', 'is_aromatic',
                        'is_isotope', 'chiral_tag', 'vocab_idx'],
                    bond_features=[
                        'bond_type', 'is_aromatic',
                        'is_conjugated', 'bond_stereo']):
        dataset = ADMETDataset(df, name, use_coords, atom_features, bond_features)
        return dataset

    def split_dataset(self, dataset):
        train, val = train_test_split(dataset, test_size=self.split_size[1] + self.split_size[2])
        train, val = train.reset_index(drop=True), val.reset_index(drop=True)
        test, val = train_test_split(val, test_size=0.5)
        test, val = test.reset_index(drop=True), val.reset_index(drop=True)
        datasets = [train, val, test]
        return datasets