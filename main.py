import sys
import torch
import argparse
from utils import open_configs
from ADMET.utils import load_models
from ADMET.train import train_ADMET
from ADMET.evaluation import evaluate
from ADMET.data import DatasetHandler

if __name__ == '__main__':
    command = sys.argv[1]    
    parser = argparse.ArgumentParser()

    if command == 'train':
        parser.add_argument('-mp', '--model_path', default=None, type=str)
        args, lf_args = parser.parse_known_args()

        # Load data & configs
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        config = open_configs(['ADMET'])[0]
        config['data']['dataset_names'] = [name.lower() for name in config['data']['dataset_names']]
        dataset_handler = DatasetHandler(**config['data']['dataset_params'])
        train_dl, val_dl, test_dl = dataset_handler.load_dataloaders(
                                        config['data']['dataset_names'],
                                        config['data']['dataloader_params'])

        root_model, head_models = load_models(
            config['data']['dataset_names'],
            args.model_path, 
            config['models']['root_model'],
            config['models']['head_models'],
            device)

        # 1st round of training
        train_ADMET(
            train_dl=train_dl, 
            val_dl=None, 
            test_dl=None,
            dataset_names=config['data']['dataset_names'],  
            epochs=config['train']['epochs'],
            val_n_steps=config['train']['val_n_steps'],
            log_n_steps=config['train']['log_n_steps'],
            optimizer_params=config['train']['optimizer_params'],
            scheduler_params=config['train']['scheduler_params'],
            root_model=root_model,
            head_models=head_models,
            device=device)

        print('Evaluating on full dataset...')
        dataset_handler.split_size = [1,0,0]
        dataloader = dataset_handler.load_dataloaders(
                                config['data']['dataset_names'],
                                config['data']['dataloader_params'])[0]

        misclassifed, _ = evaluate(
            root_model=root_model,
            head_models=head_models,
            dataloader=dataloader,
            dataset_names=config['data']['dataset_names'],
            device=device,
            output_log=True)

    elif command == 'evaluate':
        parser.add_argument('-mp', '--model_path', required=True, type=str)
        parser.add_argument('-fi', '--finetune_iters', required=True, type=str)
        parser.add_argument('-d', '--device', default=None, type=str)
        args, lf_args = parser.parse_known_args()

        if args.device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device

        config = open_configs(['ADMET'])[0]

        # Load data & configs
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')
        config = open_configs(['ADMET'])[0]
        dataset_handler = DatasetHandler(**config['data']['dataset_params'])
        dataset_handler.split_size = [1]
        dataloader = dataset_handler.load_dataloaders(
                                    config['data']['dataset_names'], 
                                    config['data']['dataloader_params'])

        root_model, head_models = load_models(
            config['data']['dataset_names'],
            args.model_path, 
            config['models']['root_model'],
            config['models']['head_models'],
            device)

        misclassifed, _ = evaluate(
            root_model=root_model,
            head_models=head_models,
            dataloader=dataloader,
            dataset_names=config['data']['dataset_names'],
            device=device,
            output_log=True)
