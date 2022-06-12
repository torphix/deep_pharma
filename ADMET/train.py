import os
from pytorch_lightning import Trainer
import torch
import datetime
import seaborn as sbn
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from ADMET.lightning import ADMETrainModule
from modules.scheduler import CustomScheduler
from torch.utils.tensorboard import SummaryWriter
from ADMET.models import HeadModel, GATNet, GCNNet
from ADMET.data import load_datasets, load_tdc_dataset


def train_ADMET(
    dataset_name,
    task_type,
    use_log_scale,
    epochs,
    val_n_steps,
    log_n_steps,
    optim_params,
    scheduler_params,
    dataset_params,
    dataloader_params,
    root_model_params,
    head_model_params,
    root_model_path=None,
    freeze_root=False,
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
    tb_logger = SummaryWriter(log_dir='logs/ADME')

    # Model
    root_model = GCNNet(**root_model_params)
    if root_model_path is not None:
        print('Loading root model...')
        root_model.load_state_dict(torch.load(root_model_path))
    head_model = nn.ModuleDict(
        {dataset_name: HeadModel()})
    if freeze_root:
        for param in root_model.parameters():
            param.requires_grad = False

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

    print(f'Root Model Parameters: {count_parameters(root_model)}M')
    print(f'Head Model Parameters: {count_parameters(head_model)}M')

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
            output = head_model[dataset_name](output)
            output = output.squeeze(1)
            batch.y = batch.y.to(device)
            # Loss 
            if use_log_scale:
                batch.y = torch.log(batch.y)
            if task_type == 'binary_classification':
                loss = F.binary_cross_entropy_with_logits(
                            output.float(), batch.y.float())
            elif task_type == 'regression':
                loss = F.mse_loss(output.float(), batch.y.float())  

            # Back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = scheduler.get_lr()

            # Validation
            if i % val_n_steps == 0:
                val_loss = validation(
                    root_model, head_model[dataset_name], val_dl, task_type, device)
                tb_logger.add_scalar('Validation Loss', val_loss)

            # Log
            if epoch % log_n_steps == 0:
                tb_logger.add_scalar('Train Loss', loss)
                tb_logger.add_scalar('Learning Rate', scheduler.get_lr())

    # Logs confusion matrix if BCE else mean error
    test_loss = test(root_model, head_model[dataset_name], test_dl, task_type, device)
    print(f'Final Train loss: {loss} Test loss: {test_loss}')
    # Save model
    print('Saving models...')
    ckpt_n = len(os.listdir('saved_models'))
    os.makedirs(f'saved_models/ckpt_{ckpt_n}/head_models')
    torch.save(root_model.state_dict(), f'saved_models/ckpt_{ckpt_n}/root_model.pth')
    torch.save(head_model.state_dict(), f'saved_models/ckpt_{ckpt_n}/head_models/{dataset_name}.pth')

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
                output = torch.sigmoid(output)
                tn, fp, fn, tp = confusion_matrix(output,  batch.y).ravel()
                # TODO plot matrix
            elif task_type == 'regression':
                loss = F.mse_loss(output.float(), batch.y.float())  
    return loss

def count_parameters(model):
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000, 3)

def progress_msg(epoch, loss, lr):
    return f'Epoch: {epoch}, Train Loss: {round(float(loss), 3)}, LR: {lr:.3e}'



def train_lightning(          
            dataset_name,
            task_type,
            log_scale):
    torch.manual_seed(42)
    train_module = ADMETrainModule(dataset_name, task_type, log_scale)

    trainer = Trainer(max_epochs=30, gpus=1)
    trainer.fit(train_module)