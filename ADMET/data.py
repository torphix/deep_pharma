import torch
import pandas as pd
from torch.utils.data import Dataset
from tdc.single_pred import ADME, Tox
from torch_geometric.data import Data, Batch
from utils import MolecularVocab, numerate_features, smiles_to_graph

class ToxicityDataset(Dataset):
    def __init__(self, dataset_name:str, label=None):
        '''
        Mapping of names to toxicity type:
        LD50_Zhu: 50% of the dose that leads to death
        hERG: Dataset of hERG blockers (hERG regulates heart beating)
        Carcinogens_Lagunin: Carcinogenic capabilities
        TODO Add more toxicity dataset types
        '''
        self.dataset = Tox(dataset_name, label_name=label)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        Data()

class ADMEDataset(Dataset):
    def __init__(self,
                df:pd.DataFrame,
                use_coords,
                atom_features,
                bond_features):
        super().__init__()
        vocab = MolecularVocab()
        df['Drug'] = df['Drug'].apply(
                smiles_to_graph, 
                node_vocab=vocab.atom_stoi,
                coords=use_coords,
                node_features=atom_features, 
                bond_features=bond_features)

        # Featurize
        src_df = pd.DataFrame(
            df['Drug'].apply(numerate_features).tolist(),
            index=df.index,
            columns=['drug', 'edge_idx', 'edge_attr'])
        self.drugs = src_df['drug']
        self.edge_idxs = src_df['edge_idx']
        self.edge_attrs = src_df['edge_attr']

        self.tgt = df['Y']

    def __len__(self):
        return len(self.tgt)

    def __getitem__(self, idx):
        data = Data(x=torch.tensor(self.drugs[idx]), 
                    edge_index=torch.tensor(self.edge_idxs[idx]),
                    edge_attr=torch.tensor(self.edge_attrs[idx]),
                    y=torch.tensor(self.tgt[idx]))
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)


def load_datasets(dataset_names):
    datasets = {}
    for name in dataset_names:
        datasets[name] = load_tdc_dataset(name)
    return datasets


def load_tdc_dataset(
        name, 
        split,
        use_coords=False,
        atom_features=[
            'atomic_mass','implicit_valence',
            'explicit_valence','formal_charge',
            'hybridization', 'is_aromatic',
            'is_isotope', 'chiral_tag', 'vocab_idx'],
        bond_features=[
            'bond_type', 'is_aromatic',
            'is_conjugated', 'bond_stereo']
        ):
    dataset = ADME(name)
    if len(split) == 1:
        dataset = ADMEDataset(dataset.get_data(), use_coords, atom_features, bond_features)
        return dataset
    else:
        split = dataset.get_split(frac=split)
        train = ADMEDataset(split['train'], use_coords, atom_features, bond_features)
        valid = ADMEDataset(split['valid'], use_coords, atom_features, bond_features)
        test = ADMEDataset(split['test'], use_coords, atom_features, bond_features)
        return train, valid, test

