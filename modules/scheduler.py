import torch

class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    '''
    Decays learning rate by gamma at each step_size interval
    until min_lr is reached at which point no further changes
    '''
    def __init__(self, optimizer, gamma, step_size, min_lr):
        self.optimizer = optimizer
        self.gamma = gamma
        self.step_size = step_size
        self.min_lr = min_lr
        self.n_steps = 0
        super().__init__(optimizer)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self.optimizer.zero_grad()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        if self.n_steps % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] > self.min_lr:
                    param_group['lr'] *= self.gamma
                elif param_group['lr'] < self.min_lr:
                    param_group['lr'] = self.min_lr
                else: continue
