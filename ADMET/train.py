import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from ADMET.evaluation import evaluate
from modules.scheduler import CustomScheduler
from torch.utils.tensorboard import SummaryWriter
from ADMET.utils import TASKS, count_parameters, progress_msg, smiles_to_fig


def train_ADMET(
    train_dl,
    val_dl,
    test_dl,
    dataset_names,
    epochs,
    val_n_steps,
    log_n_steps,
    optimizer_params,
    scheduler_params,
    root_model,
    head_models,
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

    if freeze_root:
        for param in root_model.parameters():
            param.requires_grad = False

    # Optimization
    model_params = list(root_model.parameters()) + list(head_models.parameters())
    optimizer = torch.optim.Adam(model_params, **optimizer_params)
    scheduler = CustomScheduler(optimizer, **scheduler_params)

    print(f'Root Model Parameters: {count_parameters(root_model)}M')
    print(f'Head Model Parameters: {count_parameters(head_models)}M')

    if freeze_root:
        print('Freeze root model')
        root_model = root_model.to(device)
        for param in root_model.parameters():
            param.requires_grad = False
    head_models = head_models.to(device)

    # Train Loop
    loss, current_lr, metric = 0, optimizer_params['lr'], 0
    for epoch in range(epochs):
        scheduler.step()    
        for i, batch in enumerate(tqdm(
            train_dl, progress_msg(epoch, loss, current_lr, metric), leave=False)):
            # Inference
            loss = 0
            for name, values in batch.items():
                output = root_model(
                                values.x.to(device), 
                                values.edge_index.to(device),
                                values.batch.to(device)) 
                output = head_models[name](output, values.edge_index.to(device), values.batch.to(device))
                output = output.squeeze(1)
                values.y = values.y.to(device)
                # Loss 
                use_log_scale = TASKS[name]['use_log_scale']
                task_type = TASKS[name]['task_type']
                if use_log_scale:
                    values.y = torch.log(values.y)
                if task_type == 'binary_classification':
                    output = torch.sigmoid(output)
                    loss += F.binary_cross_entropy(
                                output.float(), values.y.float()).mean()
                    output = torch.round(output) 
                elif task_type == 'regression':
                    loss += F.mse_loss(output.float(), values.y.float())  

            # Back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_lr = scheduler.get_lr()

            # Validation
            if i % val_n_steps == 0 and val_dl is not None:
                val_misclassified, val_loss = evaluate(
                    root_model,
                    head_models,
                    val_dl,
                    dataset_names,
                    device,
                    output_log=False)
                for name, smiles in val_misclassified.items():
                    figs = smiles_to_fig(smiles)
                    for fig in figs:
                        tb_logger.add_figure(f'Misclassifed Validation: {name}', fig)
                tb_logger.add_scalar('Validation Loss', val_loss)

            # Log
            if epoch % log_n_steps == 0:
                tb_logger.add_scalar('Train Loss', loss)
                tb_logger.add_scalar('Learning Rate', scheduler.get_lr())

    # Test set evaluation
    if test_dl is not None:
        print('Evaluating on Test set...')
        test_misclassified, test_loss = evaluate(
            root_model,
            head_models,
            test_dl,
            dataset_names,
            device,
            output_log=True)

        for name, smiles in test_misclassified.items():
            figs = smiles_to_fig(smiles)
            for fig in figs:
                tb_logger.add_figure(f'Misclassifed Test: {name}', fig)
        tb_logger.add_scalar('Test Loss', test_loss)

    # Save model
    print('Saving models...')
    ckpt_n = len(os.listdir('saved_models'))
    os.makedirs(f'saved_models/ckpt_{ckpt_n}/head_models')
    torch.save(root_model.state_dict(), f'saved_models/ckpt_{ckpt_n}/root_model.pth')
    for k,v in head_models.items():
        torch.save(v.state_dict(), f'saved_models/ckpt_{ckpt_n}/head_models/{k}.pth')
