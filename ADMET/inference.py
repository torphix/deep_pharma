import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ADMET.data import load_tdc_dataset
from ADMET.models import GCNNet, HeadModel


def dataset_inference(dataset_name, model_path):
    # Load models
    root_model = GCNNet(10, 1)
    root_model.load_state_dict(torch.load(f'{model_path}/root_model.pth'))

    head_model = HeadModel()
    head_model.load_state_dict(torch.load(f'{model_path}/head_models/{dataset_name}.pth'))

    # Dataset
    dataset = load_tdc_dataset(dataset_name, split=[1])
    dataloader = DataLoader(dataset, 256, shuffle=True, collate_fn=dataset.collate_fn)

    for batch in dataloader:
        output = root_model(batch.x, batch.edge_index, batch.batch)    
        output = head_model(output)
        output = output.squeeze(1)
        loss = F.mse_loss(output, batch.y)

    print(loss)

