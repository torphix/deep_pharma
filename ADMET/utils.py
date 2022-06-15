import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw
from ADMET.models import HeadModel, RootModel

def load_models(dataset_names,
                save_path,
                root_model_params,
                head_model_params,
                device):

    if save_path is not None:
        print(f'Loading models from {save_path}')
    head_models = nn.ModuleDict({
    name.lower(): HeadModel(**head_model_params)    
    for name in dataset_names})

    if save_path is not None:
        for k, model in head_models.items():
            model.load_state_dict(torch.load(f'{save_path}/head_models/{k}.pth'))

    # Load constants
    root_model = RootModel(**root_model_params)
    if save_path is not None:
        root_model.load_state_dict(torch.load(f'{save_path}/root_model.pth'))
    return root_model.to(device), head_models.to(device)
    
def select_misclassified(outputs, targets, values):
    outputs = torch.round(torch.sigmoid(outputs))
    misclassified_idxs = outputs != targets
    misclassified_idxs = misclassified_idxs.int().nonzero()
    misclassified_idxs = range(0, misclassified_idxs.shape[0])
    values = [values[idx] for idx in misclassified_idxs]
    return set(values)

def count_parameters(model):
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000, 3)

def progress_msg(epoch, loss, lr, metric):
    return f'Epoch: {epoch}, Train Loss: {round(float(loss), 3)}, LR: {lr:.3e} Metric: {metric}'

def smiles_to_fig(smiles:list):
    figs = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        figs.append(Draw.MolToMPL(mol))
    return figs

def smiles_to_img(smiles:list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    imgs = Draw.MolsToImage(mols, subImgSize=(120, 120), legends=smiles)
    return imgs

TASKS = {
    "caco2_wang": {
        "task_type": "regression",
        "use_log_scale": False,
        "label_list": False
    },
    "hia_hou": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "pgp_broccatelli": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "bioavailability_ma": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "lipophilicity_astrazeneca": {
        "task_type": "regression",
        "use_log_scale": False,
        "label_list": False
    },
    "solubility_aqsoldb": {
        "task_type": "regression",
        "use_log_scale": False,
        "label_list": False
    },
    "hydrationfreeenergy_freesolv": {
        "task_type": "regression",
        "use_log_scale": True,
        "label_list": False
    },
    "bbb_martins": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "ppbr_az": {
        "task_type": "regression",
        "use_log_scale": True,
        "label_list": False
    },
    "vdss_lombardo": {
        "task_type": "regression",
        "use_log_scale": True,
        "label_list": False
    },
    # Metabolism
    "cyp2c19_veith": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "cyp2d6_veith": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "cyp3a4_veith": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    # Toxicity
    "ames": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "herg": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "dili": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "skin reaction": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "carcinogens_lagunin": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    "tox21": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": True
    },
    "toxcast": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": True
    },
    "clintox": {
        "task_type": "binary_classification",
        "use_log_scale": False,
        "label_list": False
    },
    'ld50_zhu':{
        "task_type": "regression",
        "use_log_scale": False,
        "label_list": False
    },
}



