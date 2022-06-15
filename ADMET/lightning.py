import torch
from tqdm import tqdm
import torch.nn as nn
import pytorch_lightning as ptl
import torch.nn.functional as F
from modules.res_gcn import ResGCN
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from modules.scheduler import CustomScheduler
from ADMET.models import HeadModel, RootModel
from ADMET.data import load_datasets, load_tdc_dataset


class ADMETrainModule(ptl.LightningModule):
    def __init__(self,
                 dataset_name,
                 task_type,
                 log_scale,
                 root_model_path=None):
        super().__init__() 
        self.dataset_name = dataset_name
        self.task_type = task_type        
        self.log_scale = log_scale

        self.root_model = RootModel(10, 128)

        if root_model_path is not None:
            self.root_model.load_state_dict(torch.load(root_model_path))

        self.head_models = nn.ModuleDict({
            dataset_name: HeadModel()
        })

        self.train_ds, self.val_ds, self.test_ds = \
            load_tdc_dataset(dataset_name, split=[0.8, 0.1,0.1])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=64,
            collate_fn=self.train_ds.collate_fn)

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=64,
            collate_fn=self.train_ds.collate_fn)

    def forward(self, batch, batch_idx):
        # Inference
        output = self.root_model(
            batch.x, batch.edge_index, batch.batch)
        output = self.head_models[self.dataset_name](output)
        output = output.squeeze(1)
        if self.log_scale:
            batch.y = torch.log(batch.y)
        # Loss
        if self.task_type == 'binary_classification':
            loss = F.binary_cross_entropy_with_logits(
                output.float(), batch.y.float())
        elif self.task_type == 'regression':
            loss = F.mse_loss(output.float(), batch.y.float())        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, batch_idx)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, batch_idx)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = CustomScheduler(optimizer, 0.9, 5, 3e-6)
        return {'optimizer':optimizer, 'scheduler': scheduler}


def train_lightning(          
            dataset_name,
            task_type,
            log_scale):
    torch.manual_seed(42)
    train_module = ADMETrainModule(dataset_name, task_type, log_scale)

    trainer = Trainer(max_epochs=30, gpus=1)
    trainer.fit(train_module)