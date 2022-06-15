import torch
import pandas as pd
import gradio as gr
from torch_geometric.data import Data, Batch
from ADMET.utils import TASKS, load_models, smiles_to_img
from utils import MolecularVocab, highlight_above, numerate_features, open_configs, smiles_to_graph

# Parameters
vocab = MolecularVocab()
saved_model_path = 'saved_models/ckpt_20'
root_model = {'in_d': 10, 'out_d': 128}
config = open_configs(['ADMET'])[0]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

root_model, head_models = load_models(
                            config['data']['dataset_names'], 
                            saved_model_path,
                            config['models']['root_model'],
                            config['models']['head_models'],
                            'cpu')

def inference(inputs):
    # Process data
    smiles = [i.strip(" ") for i in inputs.split(",")]

    graphs = [smiles_to_graph(s, vocab.atom_stoi, coords=False) for s in smiles]
    features = [numerate_features(graph) for graph in graphs]
    data = [Data(x=data[0], edge_index=data[1]) for data in features]
    batch = Batch.from_data_list(data)
    root_model.eval()
    head_models.eval()
    with torch.no_grad():
        root_output = root_model(batch.x, batch.edge_index, batch.batch)
        head_outputs = {}
        for name, head in head_models.items():
            head_output = head(root_output)
            if TASKS[name]['task_type'] == 'binary_classification':
                head_output = torch.sigmoid(head_output)
            elif TASKS[name]['use_log_scale']:
                head_output = torch.exp(head_output)
            head_outputs[name] = [v[0] for v in torch.round(head_output, decimals=2).tolist()]
    # Data formatting
    df_out = pd.DataFrame(data=head_outputs).round(2)
    df_out['Molecules'] = smiles
    img = smiles_to_img(smiles)
    return df_out, img



adme_interface = gr.Interface(inference, inputs='text', outputs=[gr.DataFrame(), gr.Image(shape=(120,120))])
ui = gr.TabbedInterface([adme_interface], ['Toxicity'])
ui.launch()


'''
Example molecules:
 Morphine: CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O
 Phenobarbital (barbituate): CCC1(C(=O)NC(=O)NC1=O)C2=CC=CC=C2
 Ivermectin: CCC(C)C1C(CCC2(O1)CC3CC(O2)CC=C(C(C(C=CC=C4COC5C4(C(C=C(C5O)C)C(=O)O3)O)C)OC6CC(C(C(O6)C)OC7CC(C(C(O7)C)O)OC)OC)C)C
 Rofecoxib: O=C2OCC(=C2\c1ccccc1)\c3ccc(cc3)S(=O)(=O)C
'''