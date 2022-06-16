import torch
import argparse
import pandas as pd
import gradio as gr
from torch_geometric.data import Data, Batch
from ADMET.utils import (TASKS, load_models,
                        smiles_to_img, numerate_features,
                        open_configs, smiles_to_graph,
                        MolecularVocab)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', '--tox_path', default='saved_models/tox', type=str)
    parser.add_argument('-ad', '--adme_path', default='saved_models/adme', type=str)
    args, lf_args = parser.parse_known_args()

    # Parameters
    vocab = MolecularVocab()
    config = open_configs(['ADMET'])[0]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Toxicity
    tox_root_model, tox_head_models = load_models(
                                ['hERG', 'AMES', 'DILI', 'Carcinogens_Lagunin', 'ClinTox'], 
                                args.tox_path,
                                config['models']['root_model'],
                                config['models']['head_models'],
                                'cpu')

    def tox_inference(inputs):
        # Process data
        smiles = [i.strip(" ") for i in inputs.split(",")]

        graphs = [smiles_to_graph(s, vocab.atom_stoi, coords=False) for s in smiles]
        features = [numerate_features(graph) for graph in graphs]
        data = [Data(x=data[0], edge_index=data[1]) for data in features]
        batch = Batch.from_data_list(data)
        tox_root_model.eval()
        tox_head_models.eval()
        with torch.no_grad():
            root_output = tox_root_model(batch.x, batch.edge_index, batch.batch)
            head_outputs = {}
            for name, head in tox_head_models.items():
                head_output = head(root_output,
                                    batch.edge_index,
                                    batch.batch)
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

    # Metabolism
    adme_root_model, adme_head_models = load_models(
                                ['CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith','BBB_Martins','Pgp_Broccatelli', 'HIA_Hou'], 
                                args.adme_path,
                                config['models']['root_model'],
                                config['models']['head_models'],
                                'cpu')

    def adme_inference(inputs):
        # Process data
        smiles = [i.strip(" ") for i in inputs.split(",")]

        graphs = [smiles_to_graph(s, vocab.atom_stoi, coords=False) for s in smiles]
        features = [numerate_features(graph) for graph in graphs]
        data = [Data(x=data[0], edge_index=data[1]) for data in features]
        batch = Batch.from_data_list(data)
        adme_root_model.eval()
        adme_head_models.eval()
        with torch.no_grad():
            root_output = adme_root_model(batch.x, batch.edge_index, batch.batch)
            head_outputs = {}
            for name, head in adme_head_models.items():
                head_output = head(root_output,
                                   batch.edge_index,
                                   batch.batch)
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




    tox_interface = gr.Interface(tox_inference, inputs='text', outputs=[gr.DataFrame(), gr.Image()])
    adme_interface = gr.Interface(adme_inference, inputs='text', outputs=[gr.DataFrame(), gr.Image()])
    ui = gr.TabbedInterface([tox_interface, adme_interface], ['Toxicity', 'Metabolism'])
    ui.launch()


'''
Example molecules:
 Morphine: CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O
 Phenobarbital (barbituate): CCC1(C(=O)NC(=O)NC1=O)C2=CC=CC=C2
 Ivermectin: CCC(C)C1C(CCC2(O1)CC3CC(O2)CC=C(C(C(C=CC=C4COC5C4(C(C=C(C5O)C)C(=O)O3)O)C)OC6CC(C(C(O6)C)OC7CC(C(C(O7)C)O)OC)OC)C)C
 Rofecoxib: O=C2OCC(=C2\c1ccccc1)\c3ccc(cc3)S(=O)(=O)C
 Doxorubicin: CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O
 Paracetamol: CC(=O)NC1=CC=C(C=C1)O
'''