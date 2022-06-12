from ADMET.utils import TASKS
from ADMET.inference import dataset_inference
from ADMET.multiprocess import multi_train_ADMET
from ADMET.train import train_ADMET, train_lightning

if __name__ == '__main__':
    # command = sys.argv[1]    
    # parser = argparse.ArgumentParser()
    dataset_name = 'Lipophilicity_AstraZeneca'
    dataset_info = TASKS[dataset_name]
    dataset_params = {
        'split':[0.8, 0.1, 0.1],
        'use_coords': False,
        'atom_features':[
            'atomic_mass','implicit_valence',
            'explicit_valence','formal_charge',
            'hybridization', 'is_aromatic',
            'is_isotope', 'chiral_tag', 'vocab_idx'],
        'bond_features':[
            'bond_type', 'is_aromatic',
            'is_conjugated', 'bond_stereo']
    }
    train_ADMET(
        dataset_name, 
        dataset_info['task_type'],
        dataset_info['use_log_scale'],
        epochs=30,
        val_n_steps=5,
        log_n_steps=5,
        optim_params={'lr':1e-4, 'betas':[0.9, 0.999]},
        scheduler_params={'gamma':0.9, 'step_size':5, 'min_lr':3e-6},
        dataset_params=dataset_params,
        dataloader_params={'shuffle':True, 'batch_size':64, 'pin_memory':True},
        root_model_params={'in_d':10, 'out_d':1},
        head_model_params={'hid_ds':[1, 128, 128, 1]},
        device='cpu')

    # multi_train_ADMET(
    #     dataset_names=['Bioavailability_Ma', 'Lipophilicity_AstraZeneca'],
    #     epochs=30,
    #     val_n_steps=10,
    #     log_n_steps=5,
    #     optim_params={'lr':1e-4, 'betas':[0.9, 0.999]},
    #     scheduler_params={'gamma':0.9, 'step_size':5, 'min_lr':3e-6},
    #     dataset_params=dataset_params,
    #     dataloader_params={'shuffle':True, 'batch_size':64, 'pin_memory':True},
    #     root_model_params={'in_d':10, 'out_d':1},
    #     device=None,
    #     )
    # train_lightning(dataset_name, dataset_info['task_type'], dataset_info['use_log_scale'])



    # dataset_inference('Lipophilicity_AstraZeneca', 'saved_models/ckpt_1')