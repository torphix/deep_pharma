from multiprocessing.sharedctypes import Value
import os
import torch
import datetime
import seaborn as sbn
import torch.nn as nn
from tqdm import tqdm
from .utils import TASKS
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from ADMET.lightning import ADMETrainModule
from modules.scheduler import CustomScheduler
from torch.utils.tensorboard import SummaryWriter
from ADMET.models import HeadModel, GATNet, GCNNet
from ADMET.data import load_datasets, load_tdc_dataset


def multi_train_ADMET(
    dataset_names,
    epochs,
    val_n_steps,
    log_n_steps,
    optim_params,
    scheduler_params,
    dataset_params,
    dataloader_params,
    root_model_params,
    device=None,
    ):
    '''
    Loads and trains on the dataset_name provided saving
    the head model under the same name, the task type
    should be binary_classification or regression 
    and log_scale should be a boolean depending on 
    the magnitude of the outputs.
    '''
    # Constants
    torch.manual_seed(42)
    if device is None:
        if torch.cuda.is_available():
            print('Using GPU')
            device = 'cuda'
        else:
            print('Using CPU')
    else:
        device = device
    # tb_logger = SummaryWriter(log_dir='logs/ADME')

    # Model
    root_model = GCNNet(**root_model_params)
    head_models = nn.ModuleDict(
        {dataset_name: HeadModel() 
        for dataset_name in dataset_names})

    print(f'Root Model Parameters: {count_parameters(root_model)}M')
    print(f'Head Models Parameters: {count_parameters(head_models)}M')

    mp.set_start_method('spawn', force=True)
    root_model.share_memory()
    processes = []
    for dataset_name in dataset_names:
        process = mp.Process(target=train, args=(
            root_model,
            head_models[dataset_name], 
            dataset_name,
            epochs,
            # Options
            optim_params,
            scheduler_params,
            dataset_params,
            dataloader_params,
            device,
            # Logging
            # tb_logger,
            log_n_steps,
            val_n_steps      
        ))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    # Save model
    print('Saving models...')
    ckpt_n = len(os.listdir('saved_models'))
    os.makedirs(f'saved_models/ckpt_{ckpt_n}/head_models')
    torch.save(root_model.state_dict(), f'saved_models/ckpt_{ckpt_n}/root_model.pth')
    for k,model in head_models.items():
        torch.save(model.state_dict(), f'saved_models/ckpt_{ckpt_n}/head_models/{k}.pth')


def validation(root_model, head_model, val_dataloader, task_type, device):
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            molecule_latent = root_model(batch.x, batch.edge_index, batch.batch) 
            output = head_model(molecule_latent)
            # Loss 
            if task_type == 'binary_classification':
                loss = F.binary_cross_entropy_with_logits(
                            output.float(), batch.y.float())
            elif task_type == 'regression':
                loss = F.mse_loss(output.float(), batch.y.float())  
    return loss

def test(root_model, head_model, val_dataloader, task_type, device):
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            molecule_latent = root_model(batch.x, batch.edge_index, batch.batch) 
            output = head_model(molecule_latent)
            output = output.squeeze(1)
            # Loss 
            if task_type == 'binary_classification':
                loss = F.mse_loss(output.float(), batch.y.float())  
                # output = torch.sigmoid(output)
                # tn, fp, fn, tp = confusion_matrix(output,  batch.y).ravel()
                # TODO plot matrix
            elif task_type == 'regression':
                loss = F.mse_loss(output.float(), batch.y.float())  
    return loss

def count_parameters(model):
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000, 3)

def progress_msg(epoch, loss, lr):
    return f'Epoch: {epoch}, Train Loss: {round(float(loss), 3)}, LR: {lr:.3e}'


def train(
        # Model
        root_model,
        head_model, 
        dataset_name,
        epochs,
        # Options
        optim_params,
        scheduler_params,
        dataset_params,
        dataloader_params,
        device,
        # Logging
        # tb_logger,
        log_n_steps,
        val_n_steps):
    task_type = TASKS[dataset_name]['task_type']
    use_log_scale = TASKS[dataset_name]['use_log_scale']

    # Optimization
    model_params = list(root_model.parameters()) + list(head_model.parameters())
    optimizer = torch.optim.Adam(model_params, **optim_params)
    scheduler = CustomScheduler(optimizer, **scheduler_params)

    # Data
    train_ds, val_ds, test_ds = load_tdc_dataset(dataset_name, **dataset_params)
    train_dl, val_dl, test_dl = (
        DataLoader(train_ds, **dataloader_params, collate_fn=train_ds.collate_fn),
        DataLoader(val_ds, shuffle=False, batch_size=64, collate_fn=val_ds.collate_fn),
        DataLoader(test_ds, shuffle=False, batch_size=64, collate_fn=test_ds.collate_fn))  

    root_model = root_model.to(device)
    head_model = head_model.to(device)

    # Train Loop
    loss, current_lr = 0, optim_params['lr']
    # progress_bar = tqdm(train_dl)
    for epoch in range(epochs):
        scheduler.step()    
        for i, batch in enumerate(tqdm(
            train_dl, progress_msg(epoch, loss, current_lr), leave=False)):
            # Inference
            output = root_model(
                                batch.x.to(device), 
                                batch.edge_index.to(device),
                                batch.batch.to(device)) 
            output = head_model(output)
            output = output.squeeze(1)
            batch.y = batch.y.to(device)
            # Loss 
            if use_log_scale:
                batch.y = torch.log(batch.y)
            if task_type == 'binary_classification':
                loss = F.mse_loss(
                            output.float(), batch.y.float())
            elif task_type == 'regression':
                loss = F.mse_loss(output.float(), batch.y.float())  
            else:
                raise ValueError(f'task_type {task_type} not found')

            # Back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = scheduler.get_lr()

    
            # # Validation
            # if i % val_n_steps == 0:
            #     val_loss = validation(
            #         root_model, head_model[dataset_name], val_dl, task_type, device)
            #     tb_logger.add_scalar('Validation Loss', val_loss)

            # # Log
            # if epoch % log_n_steps == 0:
            #     tb_logger.add_scalar('Train Loss', loss)
            #     tb_logger.add_scalar('Learning Rate', scheduler.get_lr())

    # Logs confusion matrix if BCE else mean error
    test_loss = test(root_model, head_model, test_dl, task_type, device)
    print(f'Final Train loss: {loss} Test loss: {test_loss} for dataset {dataset_name}')

