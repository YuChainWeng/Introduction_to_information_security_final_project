import os
import sys
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor
from copy import deepcopy
from torch._C import device
from torch.optim import SGD, Adam
from tqdm.auto import tqdm

from torch_geometric.transforms import ToSparseTensor
from torch_geometric.data import Data

import torch.nn as nn
import torch.nn.functional as F
from estimator import EstimateAdj, PGD, prox_operators
import time

def run_one_epoch(trainer, *args, **kwargs):
    """Wrapper to time a single call to ``train_adj``.
    Parameters are forwarded directly to ``trainer.train_adj``.
    Returns the duration in seconds and peak GPU memory in MB.
    """
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train_adj(*args, **kwargs)
    torch.cuda.synchronize()
    dur = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1e6
    return dur, peak

class Trainer:
    def __init__(
            self,
            optimizer:      dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
            max_epochs:     dict(help='maximum number of training epochs') = 500,
            learning_rate:  dict(help='learning rate') = 0.01,
            weight_decay:   dict(help='weight decay (L2 penalty)') = 0.0,
            patience:       dict(help='early-stopping patience window size') = 0,
            orphic:         dict(help='use l1 norm', option='-pro') = False,
            inner_epoch:    dict(help='change estimator training ratio', option='-inner') = 2,
            device='cuda',
            logger=None,
            alpha = 5e-4,
            gamma = 1,
        ):
        self.optimizer_name = optimizer
        self.max_epochs = max_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.pro = orphic
        self.inner_epoch = inner_epoch
        self.logger = logger
        self.model = None
        self.alpha = alpha
        self.gamma = gamma
        self.estimator = None
        

    def configure_optimizers(self):
        if self.optimizer_name == 'sgd':
            return SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, model, data, args):

        self.model = model.to(self.device)
        data = data.to(self.device)
        optimizer = self.configure_optimizers()
        num_epochs_without_improvement = 0
        best_metrics = None
        
        # Load a model checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f'{args.dataset, args.x_eps, args.e_eps, args.x_steps, args.y_steps, args.model}.pt')
        if os.path.exists(ckpt_path) and not args.retrain:
            print("Load checkpoint from {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            model.eval()
            val_metrics = self._validation(data)
            return val_metrics
        else:
            args.retrain = True

        # Retrain
        epoch_progbar = tqdm(range(1, self.max_epochs + 1), desc='Epoch: ', leave=False, position=1, file=sys.stdout)
        for epoch in epoch_progbar:
            metrics = {'epoch': epoch}
            train_metrics = self._train(data, optimizer, epoch)
            metrics.update(train_metrics)

            val_metrics = self._validation(data)
            metrics.update(val_metrics)
            ''' start using pro-gnn loss function'''

            if self.pro:
                for i in range(self.inner_epoch):
                    num_nodes = data.num_nodes
                    device = data.x.device
                    perturbed_adj = to_torch_coo_tensor(
                        data.edge_index,
                        torch.ones(data.edge_index.size(1), device=device, dtype=torch.float32),
                        size=(num_nodes, num_nodes)
                    ).coalesce() 
                    estimator = EstimateAdj(perturbed_adj, symmetric=False, device=self.device).to(self.device)
                    self.estimator = estimator
                    self.optimizer_adj = SGD(estimator.parameters(), momentum=0.9, lr=self.learning_rate)
                    self.optimizer_l1 = PGD(estimator.parameters(), proxs=[prox_operators.prox_l1], lr=self.learning_rate, alphas=[self.alpha])
                    self.train_adj(epoch=epoch, i=i, features=data.x, adj=perturbed_adj, data=data)
                dur, peak = run_one_epoch(
                    self,
                    epoch=epoch,
                    i=i,
                    features=data.x,
                    adj=perturbed_adj,
                    data=data
                )
                print(f'Time/epoch: {dur:.3f}s,  Peak GPU: {peak:.1f} MB')


            if self.logger:
                self.logger.log(metrics)

            if best_metrics is None or (
                metrics['val/loss'] < best_metrics['val/loss'] and
                best_metrics['val/acc'] < metrics['val/acc'] <= metrics['train/maxacc'] and
                best_metrics['train/acc'] < metrics['train/acc'] <= 1.05 * metrics['train/maxacc']
            ):
                best_metrics = metrics
                num_epochs_without_improvement = 0
                
                torch.save({
                    'model': deepcopy(model.state_dict()),
                    'optimizer': deepcopy(optimizer.state_dict()),
                    'epoch': epoch
                }, ckpt_path)

            else:
                num_epochs_without_improvement += 1
                if num_epochs_without_improvement >= self.patience > 0:
                    break

            # display metrics on progress bar
            epoch_progbar.set_postfix(metrics)
        
        if self.logger:
            self.logger.log_summary(best_metrics)
        
        return best_metrics
    

    def _train(self, data, optimizer, i):
        self.model.train()
        optimizer.zero_grad()
        
        if self.pro and i!=1:
            data = self.update_data(data)

        loss, metrics = self.model.training_step(data)
        loss.backward()
        optimizer.step()
        return metrics

    @torch.no_grad()
    def _validation(self, data):
        self.model.eval()
        return self.model.validation_step(data)

    def update_data(self, data):
        A_sp      = self.estimator() 
        edge_index = A_sp.indices() 
        # package clean adj matrix
        data_ = Data(T=data.T, edge_index=edge_index, test_mask=data.test_mask, train_mask=data.train_mask, val_mask=data.val_mask, x=data.x, y=data.y)
        data_ = ToSparseTensor(remove_edge_index=False)(data_)
        data_.name = data.name
        data_.num_classes = data.num_classes
        return data_.to(self.device)

    def train_adj(self, epoch, i, features, adj, data=None):
        estimator = self.estimator
        t_ = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        num_nodes = data.num_nodes
        device = data.x.device
        A_sp = to_torch_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.size(1), device=device, dtype=torch.float32),
            size=(num_nodes, num_nodes)
        ).coalesce()
        A_est_sp = estimator()

        loss_fro = torch.pow((A_est_sp - A_sp).coalesce().values(), 2).sum()
        loss_l1 = A_est_sp.values().abs().sum()
        
        _, _, p_yt_x = self.model(data)
        loss_gnn = self.model.cross_entropy_loss(p_y=p_yt_x[data.train_mask], y=self.model.cached_yt[data.train_mask], weighted=False)

        # use total loss
        # total_loss = loss_fro + self.gamma * loss_gnn + self.alpha * loss_l1 
        # total_loss.backward()

        loss_diffiential =  loss_fro + self.gamma * loss_gnn 
        loss_diffiential.backward()

        self.optimizer_adj.step()

        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()

        with torch.no_grad():
            estimator.values.data.clamp_(0.0, 1.0)

        self.model.eval()

        print(
            'Epoch_Pro: {:04d}_{:d}'.format(epoch+1, i),
            'diffiential loss: {:.4f}'.format(loss_diffiential),
            'time: {:.4f}s'.format(time.time() - t_)
        )