import re
import yaml
import torch
import numpy as np
import selfies as sf
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.utils import from_scipy_sparse_matrix


class MolecularVocab():
    def __init__(self, src_file=None):
        '''Creates a smiles vocab or loads default one'''
        if src_file is None:
            self.selfies_vocab_stoi = self.load_default_selfies_vocab()

        self.selfies_vocab_itos = {v: i for i,
                                   v in self.selfies_vocab_stoi.items()}

        _atom_vocab = {
            'Dy', 'Hf', 'Pd', 'P', 'Ni', 'Cs', 'H', 'Bi', 'O', 'Na', 'Gd',
            'Mn', 'Ca', 'La', 'F', 'Lu', 'Fe', 'Zn', 'I', 'S', 'Cd', 'N',
            'In', 'Ta', 'Pt', 'Y', 'Ce', 'Au', 'Se', 'K', 'Pr', 'Rh', 'C', 
            'Cr', 'As', 'V', 'Br', 'Si', 'Sb', 'Mo', 'Te', 'Hg', 'Li', 'Ag', 
            'Ru', 'Be', 'Pb', 'Al', 'Zr', 'Nb', 'W', 'Re', 'Sn', 'Ge', 'Sr',
            'Nd', 'Ti', 'Cu', 'Ir', 'Ba', 'Cl', 'Sm', 'Mg', 'B', 'Co','Tl', 'Tc', '*'}
        self.atom_stoi = {v:i for i,v in enumerate(_atom_vocab)}
        self.atom_itos = {i:v for i,v in enumerate(_atom_vocab)}

    def smiles_stoi(self, smiles_string):
        selfies = sf.encoder(smiles_string)
        return self.selfies_stoi(selfies)

    def smiles_itos(self, idxs):
        selfies = self.selfies_itos(idxs)
        selfies = "".join(selfies)
        return sf.decoder(selfies)

    def selfies_stoi(self, selfie_string):
        values = list(sf.split_selfies(selfie_string))
        idxs = [self.selfies_vocab_stoi[v] for v in values]
        return idxs

    def selfies_itos(self, idxs):
        values = [self.selfies_vocab_itos[idx] for idx in idxs]
        return values

    def tokenize(self, file):
        regex_vocab = re.compile(
            r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p
                |\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*
                |\$|\%[0-9]{2}|[0-9])"""
        )
        with open(file, 'r') as f:
            text = f.read()
        return [t for t in regex_vocab.findall(text)]

    def load_default_selfies_vocab(self):
        return {
            '<pad>': 0, '<eos>': 1, '<sos>': 2,
            '[#Branch1]': 3, '[#Branch2]': 4, '[#C-1]': 5,
            '[#C]': 6, '[#N+1]': 7, '[#N]': 8, '[=As]': 9,
            '[=Branch1]': 10, '[=Branch2]': 11, '[=C]': 12,
            '[=N+1]': 13, '[=N-1]': 14, '[=NH1+1]': 15,
            '[=NH2+1]': 16, '[=N]': 17, '[=O+1]': 18, '[=O]': 19,
            '[=P+1]': 20, '[=PH1]': 21, '[=P]': 22, '[=Ring1]': 23,
            '[=Ring2]': 24, '[=S+1]': 25, '[=SH1]': 26, '[=S]': 27,
            '[=Se+1]': 28, '[=Se]': 29, '[=Te+1]': 30, '[Al]': 31,
            '[As]': 32, '[B-1]': 33, '[BH1-1]': 34, '[BH2-1]': 35,
            '[BH3-1]': 36, '[B]': 37, '[Br]': 38, '[Branch1]': 39,
            '[Branch2]': 40, '[C+1]': 41, '[C-1]': 42, '[CH1-1]': 43,
            '[C]': 44, '[Cl+3]': 45, '[Cl]': 46, '[F]': 47, '[H]': 48,
            '[I+1]': 49, '[I]': 50, '[N+1]': 51, '[N-1]': 52,
            '[NH1+1]': 53, '[NH1-1]': 54, '[NH1]': 55, '[NH2+1]': 56,
            '[N]': 57, '[Na]': 58, '[O+1]': 59, '[O-1]': 60,
            '[OH0]': 61, '[OH1+1]': 62, '[O]': 63, '[P+1]': 64,
            '[PH1]': 65, '[P]': 66, '[Ring1]': 67, '[Ring2]': 68,
            '[S+1]': 69, '[S-1]': 70, '[SH1]': 71, '[S]': 72,
            '[Se+1]': 73, '[SeH1]': 74, '[Se]': 75, '[Si]': 76, '[Te]': 77}
            

'''Graph Generation'''
def smiles_to_graph(smiles: str, 
                    node_vocab=None,
                    coords=None,
                    node_features=[
                        'atomic_mass','implicit_valence',
                        'explicit_valence','formal_charge',
                        'hybridization', 'is_aromatic',
                        'is_isotope', 'chiral_tag', 'vocab_idx'
                    ], 
                    bond_features=[
                        'bond_type', 'is_aromatic',
                        'is_conjugated', 'bond_stereo',
                    ]):
    graph = init_graph(smiles, coords=coords, node_vocab=node_vocab)
    graph = add_atoms_to_graph(graph, node_features)
    graph = add_bonds_to_graph(graph, bond_features)
    return graph

def init_graph(smiles, node_vocab, coords=False):
    '''
    coords: if True computes 3D coords
    Vocab: 
        stoi vocabulary dict for indexing char
    '''
    if coords:
        mol = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol)
        conformer = mol.GetConformer()
        coords = conformer.GetPositions()
    return nx.Graph(
        smiles=smiles,
        rdmol=Chem.MolFromSmiles(smiles),
        node_vocab=node_vocab,
        coords=coords)
    
def add_atoms_to_graph(graph: nx.Graph, atom_features: list):
    for i, atom in enumerate(graph.graph['rdmol'].GetAtoms()):
        # Calculate features
        attr_dict = {}
        if 'atomic_mass' in atom_features:
            attr_dict['atomic_mass'] = atom.GetMass()
        if 'implicit_valence' in atom_features:
            attr_dict['implicit_valence'] = atom.GetImplicitValence()
        if 'explicit_valence' in atom_features:
            attr_dict['explicit_valence'] = atom.GetExplicitValence()
        if 'formal_charge' in atom_features:
            attr_dict['formal_charge'] = atom.GetFormalCharge()
        if 'hybridization' in atom_features:
            attr_dict['hybridization'] = atom.GetHybridization()
        if 'is_aromatic' in atom_features:
            attr_dict['is_aromatic'] = atom.GetIsAromatic()
        if 'is_isotope' in atom_features:
            attr_dict['is_isotope'] = atom.GetIsotope()
        if 'chiral_tag' in atom_features:
            attr_dict['chiral_tag'] = atom.GetChiralTag()
        if graph.graph['coords']:
            attr_dict['coords'] = graph.graph['coords'][i]

        if graph.graph['node_vocab'] is not None:
            attr_dict['vocab_idx'] = graph.graph['node_vocab'][atom.GetSymbol()]

        # Add node
        graph.add_node(
            f"{atom.GetSymbol()}:{atom.GetIdx()}",
            atom_type=atom.GetSymbol(),
            atom_idx=atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            rdmol_atom=atom,
            **attr_dict
        )
    return graph

def add_bonds_to_graph(graph: nx.Graph, bond_features:list):
    bond_attrs = dict()
    for i, bond in enumerate(graph.graph['rdmol'].GetBonds()):
        # Calculate bond data
        if 'bond_type' in bond_features:
            bond_attrs['bond_type'] = bond.GetBondType()
        if 'is_aromatic' in bond_features:
            bond_attrs['is_aromatic'] = bond.GetIsAromatic()
        if 'is_conjugated' in bond_features:
            bond_attrs['is_conjugated'] = bond.GetIsConjugated()
        if 'bond_stereo' in bond_features:
            bond_attrs['bond_stereo'] = bond.GetStereo()
        # Add bonds
        start_node = f'{bond.GetBeginAtom().GetSymbol()}:{bond.GetBeginAtomIdx()}'
        end_node = f'{bond.GetEndAtom().GetSymbol()}:{bond.GetEndAtomIdx()}'
        graph.add_edge(
            start_node, end_node, **bond_attrs)
    return graph


def numerate_features(
    graph:nx.Graph, 
    skip_node_features=['atomic_mass','rdmol_atom', 'vocab_idx', 'atom_idx']):
    '''
    Iterates over node and edge features converting 
    to numerical format for input into GNN's
    Skip features specifies which features to not include
    in the featurization 
    (atom_type is skipped as atomic mass is already present)
    Note if when featurizing 3D coords it is done so in a 
    non deterministic manner
    For class values such as atom_type and hybridization type
    values are one hot vectors concated to the main feature vector
    '''
    hybridization_dict = {
        Chem.rdchem.HybridizationType.SP3: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP: 2,
        Chem.rdchem.HybridizationType.S: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 6,
    }
    stereo_dict = {
        Chem.rdchem.BondStereo.STEREOANY: 0,
        Chem.rdchem.BondStereo.STEREOCIS: 1,
        Chem.rdchem.BondStereo.STEREOE: 2,
        Chem.rdchem.BondStereo.STEREONONE: 3,
        Chem.rdchem.BondStereo.STEREOTRANS: 4,
        Chem.rdchem.BondStereo.STEREOZ: 5,
    }
    bond_dict = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    # Remove skip items
    graph = remove_attr_from_nodes(graph, skip_node_features)
    # Featurize atoms
    node_features = []
    for node_idx, node in enumerate(graph.nodes(data=True)):
        node, data = node[0], node[1]
        for i, (k, v) in enumerate(data.items()):
            if 'vocab_idx' == k:
                atom_one_hot = torch.zeros((12))
                atom_one_hot[v] = 1
                node_features.append(atom_one_hot)
            elif 'implicit_valence' == k:
                node_features.append(torch.tensor([v]))
            elif 'formal_charge' == k:
                node_features.append(torch.tensor([v]))
            elif 'hybridization' == k:
                hyb_type = hybridization_dict[v]
                hyb_one_hot = torch.zeros((len(hybridization_dict.keys())))
                hyb_one_hot[hyb_type] = 1
                node_features.append(hyb_one_hot)
            elif 'is_aromatic' == k:
                node_features.append(torch.tensor([float(v)]) + 1)
            elif 'is_isotope' == k:
                node_features.append(torch.tensor([float(v)]) + 1)
            elif 'coords' == k:
                node_features.append(torch.tensor([v]))
    nodes = torch.cat(node_features).view(len(graph.nodes), -1)
    # Edge features
    # edge_attrs = []
    # for edge_idx, edge in enumerate(graph.edges(data=True)):
    #     _, _, node_data = edge
    #     for i, (k, v) in enumerate(node_data.items()):
    #         if k == 'bond_type':
    #             edge_one_hot = torch.zeros((len(bond_dict.keys())))
    #             edge_one_hot[bond_dict[v]] = 1
    #             edge_attrs.append(edge_one_hot)
    #         elif k == 'bond_stereo': 
    #             stereo_one_hot = torch.zeros((len(stereo_dict.keys())))
    #             stereo_one_hot[stereo_dict[v]] = 1
    #             edge_attrs.append(stereo_one_hot)
    #         else:
    #             edge_attrs.append(torch.tensor([float(v)]) + 1)
    # edge_attrs = torch.cat(edge_attrs).view(len(graph.edges()), -1)
    # Adj matrix
    adj_matrix = nx.adjacency_matrix(graph)
    adj_matrix = from_scipy_sparse_matrix(adj_matrix)[0]
    return (torch.tensor(nodes), adj_matrix) #torch.tensor(edge_attrs))

def remove_attr_from_nodes(graph:nx.Graph, attrs:list):
    for node, data in graph.nodes(data=True):
        for attr in attrs:
            del graph.nodes[node][attr]
    return graph

def open_configs(configs:list):
    opened_configs = []
    for config in configs:
        with open(f'configs/{config}.yaml', 'r') as f:
            opened_configs.append(yaml.load(f.read(), Loader=yaml.FullLoader))
    return opened_configs


def write_file(file, contents):
    with open(file, 'w') as f:
        f.write(contents)

# Pandas styling
def highlight_above(s, props='color:white;background-color:red'):
    return np.where(s > 0.6, props, '')