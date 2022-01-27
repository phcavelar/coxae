import numpy as np
import sklearn
import lifelines

import torch
import torch.nn as nn
import torch.nn.functional as F
import pycox

from .base import SurvivalClustererMixin
from .architectures import CoxAutoencoder

__DEFAULT_AE_KWARGS = {
    "hidden_dims": [512],
    "encoding_dim": 128,
    "cox_hidden_dims": [],
    "nonlinearity": F.relu,
    "final_nonlinearity": lambda x:x,
    "dropout_rate": 0.3,
    "bias": True,
}

__DEFAULT_AE_OPT_KWARGS = {
    "lr":1e-3,
    "weight_decay": 1e-4
}

__DEFAULT_AE_TRAIN_KWARGS = {
    "epochs":256,
    "noise_std": 0.2
}

class CoxAutoencoderClustering(SurvivalClustererMixin):

    def __init__(self,
            *args,
            n_clusters=2,
            d_in:int = None,
            ae_kwargs:dict = __DEFAULT_AE_KWARGS,
            ae_opt_kwargs = __DEFAULT_AE_OPT_KWARGS,
            ae_train_opts = __DEFAULT_AE_TRAIN_KWARGS,
            **kwargs):
        super().__init__(*args, n_clusters=n_clusters, **kwargs)
        
        self.ae_kwargs = ae_kwargs
        if d_in is not None:
            self.__init_ae(d_in, **ae_kwargs)
            self.ae_initialised = True
        else:
            self.ae_initialised = False

        self.ae_opt_kwargs = ae_opt_kwargs
        if self.ae_opt_kwargs is not None:
            self.__init_ae_opt(**ae_opt_kwargs)
            self.ae_opt_initialised = True
        else:
            self.ae_opt_initialised = False
        
        self.ae_train_opts = ae_train_opts

    # Mixin functions

    def fit(self, X: np.array, durations: np.array, events: np.array, *args, ae_opt_kwargs=None, ae_train_opts=None, **kwargs):
        if not self.ae_initialised:
            self.__init_ae(X.shape[-1], **self.ae_kwargs)
        if not self.ae_opt_initialised:
            self.ae_opt_kwargs = self.ae_opt_kwargs if ae_opt_kwargs is not None else ae_opt_kwargs
            self.__init_ae_opt(**self.ae_opt_kwargs)
        self.ae_train_opts = self.ae_train_opts if ae_train_opts is not None else ae_train_opts
        self.__train_ae(X, durations, events, *args, **self.ae_train_opts)
        raise NotImplementedError()
    
    def predict(self, X: np.array, *args, **kwargs):
        raise NotImplementedError()


    # AE-related functions

    def __init_ae(self, input_dim:int,
                hidden_dims:list = __DEFAULT_AE_KWARGS["hidden_dims"],
                encoding_dim:int = __DEFAULT_AE_KWARGS["encoding_dim"],
                cox_hidden_dims:list = __DEFAULT_AE_KWARGS["cox_hidden_dims"],
                nonlinearity:function = __DEFAULT_AE_KWARGS["nonlinearity"],
                final_nonlinearity:function = __DEFAULT_AE_KWARGS["final_nonlinearity"],
                dropout_rate:float = __DEFAULT_AE_KWARGS["dropout_rate"],
                bias:bool = __DEFAULT_AE_KWARGS["bias"],
                **kwargs) -> None:
        self.ae = CoxAutoencoder(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            encoding_dim = encoding_dim,
            cox_hidden_dims = cox_hidden_dims,
            nonlinearity = nonlinearity,
            final_nonlinearity = final_nonlinearity,
            dropout_rate = dropout_rate,
            bias = bias, 
            **kwargs
        )

    def __init_ae_opt(self, lr=__DEFAULT_AE_OPT_KWARGS["lr"], weight_decay=__DEFAULT_AE_OPT_KWARGS["weight_decay"], **kwargs) -> None:
        self.opt = torch.optim.Adam(
            self.ae.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    
    def __train_ae(self, X: np.array, durations: np.array, events: np.array, *args, epochs=__DEFAULT_AE_TRAIN_KWARGS["epochs"], noise_std=__DEFAULT_AE_TRAIN_KWARGS["noise_std"], **kwargs) -> None:
        
        tX_clean = torch.tensor(X, dtype=torch.float32)
        tT = torch.tensor(durations, dtype=torch.float32)
        tE = torch.tensor(events)

        losses = []
        self.ae.train()
        for e in range(epochs):
            self.opt.zero_grad()
            
            tX_noise = tX_clean + torch.normal(torch.zeros_like(tX_clean), noise_std)
            
            y = self.ae(tX_noise)
            
            reconstruction_loss = F.mse_loss(y, tX_clean)
            
            cox_loss = pycox.models.loss.cox_ph_loss(self.ae.cox(tX_clean), tT, tE)
            
            loss = reconstruction_loss + cox_loss
            
            loss.backward()
            self.opt.step()

            losses.append(loss.detach().numpy().item())

    def integrate_ae(self,X):
        self.ae.eval()
        return self.ae.encode(torch.tensor(X, dtype=torch.float32)).detach().numpy()

    # Cox-Selection related inputs
    def get_significant_factors_indexes(self, x):
        raise NotImplementedError()