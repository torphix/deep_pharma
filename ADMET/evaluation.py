import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from .utils import TASKS, select_misclassified


def evaluate(
    root_model,
    head_models,
    dataloader,
    dataset_names,
    device,
    output_log):
    '''
    Iterates over data extracting and returning the misclassified
    and the cumulative loss, output_log prints precision and recall
    '''
    tn, fp, fn, tp = 0, 0, 0, 0
    misclassifed = {name: [] for name in dataset_names}
    root_model = root_model.eval()
    head_models = head_models.eval()
    with torch.no_grad():
        loss = 0
        for batch in dataloader:
            for name, values in batch.items():
                output = root_model(
                                values.x.to(device), 
                                values.edge_index.to(device),
                                values.batch.to(device)) 
                output = head_models[name](output)
                output = output.squeeze(1)
                values.y = values.y.to(device)
                # Loss 
                use_log_scale = TASKS[name]['use_log_scale']
                task_type = TASKS[name]['task_type']
                if use_log_scale:
                    values.y = torch.log(values.y)
                if task_type == 'binary_classification':
                    loss += F.binary_cross_entropy_with_logits(
                                output.float(), values.y.float()).mean()
                    values.y = values.y.detach().cpu()
                    output = torch.round(torch.sigmoid(output)).detach().cpu() 
                    tnc, fpc, fnc, tpc = confusion_matrix(values.y,  output).ravel()
                    tn += tnc 
                    fp += fpc
                    fn += fnc
                    tp += tpc
                    total_misclassified = misclassifed[name]
                    misclassifed[name] = set(total_misclassified) | select_misclassified(output, values.y, values.smiles)
                elif task_type == 'regression':
                    loss += F.mse_loss(output.float(), values.y.float()) 
    if output_log:
        print(f'TN: {tn} FP: {fp}, FN: {fn}, TP: {tp}')
        print(f'Precision: {tp/(tp+fp)} Recall: {tp/(tp+fn)}')
        print(f'Total misclassified: {sum([len(v) for v in misclassifed.values()])}')
    return misclassifed, loss / len(dataloader)

