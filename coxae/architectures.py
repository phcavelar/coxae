import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            nonlinearity,
            dropout_rate=0.5,
            bias=True,
            ):
        super().__init__()
        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [output_dim]
        
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out, bias=bias) for d_in, d_out in zip(in_dims, out_dims)])
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(self.nonlinearity(layer(x)))
        return self.layers[-1](x)
    
    def layer_activations(self, x):
        # To allow for activation normalisation
        activations = [x]
        for layer in self.layers[:-1]:
            activations.append(self.dropout(self.nonlinearity(layer(activations[-1]))))
        return activations[1:] + [self.layers[-1](activations[-1])]

class ConcreteSelect(nn.Module):
    def __init__(
            self,
            input_dim:int,
            output_dim:int,
            start_temp:float = 10.0,
            min_temp:float = 0.1,
            alpha:float = 0.99999,
            eps:float = 1e-6,
            init_fn = lambda x: nn.init.xavier_uniform_(x, gain=1.0),
            ):
        super().__init__()
        self.eps = eps
        self.min_temp = torch.tensor(min_temp, requires_grad=False)
        self.alpha = torch.tensor(alpha, requires_grad=False)
        logits = torch.empty((input_dim, output_dim))
        init_fn(logits)
        self.logits = nn.parameter.Parameter(logits)
        self.temp = torch.tensor(start_temp, requires_grad=False)
        
    def forward(self, X):
        if not self.training:
            return X[..., torch.argmax(self.logits, dim=0)]
        samples = self.get_alphas(X)
        return torch.matmul(X, samples)
    
    def update_temperature(self):
        self.temp.set_(max(self.min_temp, self.temp * self.alpha))
    
    def get_alphas(self, X):
        if not self.training:
            return F.softmax(self.logits, dim=0) / self.temp
        # Generate a uniform distribution with a minimum of eps (to avoid numerical errors) and 1
        uniform = torch.empty(self.logits.shape)
        uniform.uniform_(self.eps, 1.0)
        # Transform it into the gumbel distribution
        gumbel = -torch.log(-torch.log(uniform))
        # Get the noisy logits
        noisy_logits = (self.logits + gumbel) / self.temp
        # Apply softmax to the values on the input axis (all input value weights' sum to one)
        return F.softmax(noisy_logits, dim=0)

class Autoencoder(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            encoding_dim,
            nonlinearity=F.relu,
            final_nonlinearity=lambda x:x,
            dropout_rate=0.5,
            bias=True,
            ):
        super().__init__()
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]
        
        self.encoder = MLP(input_dim, hidden_dims, encoding_dim, nonlinearity, dropout_rate, bias)
        self.decoder = MLP(encoding_dim, hidden_dims[-1::-1], input_dim, nonlinearity, dropout_rate, bias)
        self.final_nonlinearity = final_nonlinearity
    
    def encode(self,x):
        return self.encoder(x)
    
    def decode(self,x):
        return self.final_nonlinearity(self.decoder(x))
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def layer_activations(self,x):
        # To allow for activation normalisation
        encoder_activations = self.encoder.layer_activations(x)
        decoder_activations = self.decoder.layer_activations(encoder_activations[-1])
        return encoder_activations + decoder_activations
    
    def get_feature_importance_matrix(self):
        with torch.no_grad():
            feature_importance_matrix = self.encoder.layers[0].weight.T
            for layer in self.encoder.layers[1:]:
                feature_importance_matrix = torch.matmul(feature_importance_matrix, layer.weight.T)
        return feature_importance_matrix.detach().cpu().numpy()

class ConcreteAutoencoder(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            encoding_dim,
            nonlinearity=F.relu,
            final_nonlinearity=lambda x:x,
            dropout_rate=0.5,
            bias=True,
            start_temp:float = 10.0,
            min_temp:float = 0.1,
            alpha:float = 0.99999,
            ):
        super().__init__()
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]
        
        self.encoder = ConcreteSelect(input_dim, encoding_dim, start_temp=start_temp, min_temp=min_temp, alpha=alpha)
        self.decoder = MLP(encoding_dim, hidden_dims[-1::-1], input_dim, nonlinearity, dropout_rate, bias)
        self.final_nonlinearity = final_nonlinearity
    
    def encode(self,x):
        return self.encoder(x)
    
    def decode(self,x):
        return self.final_nonlinearity(self.decoder(x))
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def layer_activations(self,x):
        # To allow for activation normalisation
        decoder_activations = self.decoder.layer_activations(self.encoder(x))
        return decoder_activations
    
    def get_feature_importance_matrix(self):
        with torch.no_grad():
            feature_importance_matrix = self.encoder.get_alphas()
        return feature_importance_matrix.detach().cpu().numpy()

    def get_selected_feature_indexes(self):
        with torch.no_grad():
            return torch.argmax(self.encoder.logits, dim=0).detach().cpu().numpy()
    
    def update_temperature(self):
        self.encoder.update_temperature()

class CoxAutoencoder(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            encoding_dim,
            cox_hidden_dims,
            nonlinearity=F.relu,
            final_nonlinearity=lambda x:x,
            dropout_rate=0.5,
            bias=True,
            ):
        super().__init__()
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]
        
        self.ae = Autoencoder(input_dim, hidden_dims, encoding_dim, nonlinearity, final_nonlinearity, dropout_rate, bias)
        self.cox_mlp = MLP(encoding_dim, cox_hidden_dims, 1, nonlinearity, dropout_rate, bias)
    
    def encode(self,x):
        return self.ae.encode(x)
    
    def decode(self,x):
        return self.ae.decode(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def cox(self,x):
        return self.cox_mlp(self.encode(x))
    
    def layer_activations(self,x):
        # To allow for activation normalisation
        encoder_activations = self.encoder.layer_activations(x)
        decoder_activations = self.decoder.layer_activations(encoder_activations[-1])
        return encoder_activations + decoder_activations
    
    def get_feature_importance_matrix(self):
        return self.ae.get_feature_importance_matrix()

class ConcreteCoxAutoencoder(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            encoding_dim,
            cox_hidden_dims,
            nonlinearity=F.relu,
            final_nonlinearity=lambda x:x,
            dropout_rate=0.5,
            bias=True,
            start_temp:float = 10.0,
            min_temp:float = 0.1,
            alpha:float = 0.99999,
            ):
        super().__init__()
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]
        
        self.ae = ConcreteAutoencoder(input_dim, hidden_dims, encoding_dim, nonlinearity, final_nonlinearity, dropout_rate, bias, start_temp=start_temp, min_temp=min_temp, alpha=alpha)
        self.cox_mlp = MLP(encoding_dim, cox_hidden_dims, 1, nonlinearity, dropout_rate, bias)
    
    def encode(self,x):
        return self.ae.encode(x)
    
    def decode(self,x):
        return self.ae.decode(x)
    
    def forward(self, x):
        return self.decode(self.encode(x))
    
    def cox(self,x):
        return self.cox_mlp(self.encode(x))
    
    def layer_activations(self,x):
        # To allow for activation normalisation
        encoder_activations = self.encoder.layer_activations(x)
        decoder_activations = self.decoder.layer_activations(encoder_activations[-1])
        return encoder_activations + decoder_activations
    
    def get_feature_importance_matrix(self):
        return self.ae.get_feature_importance_matrix()

    def get_selected_feature_indexes(self):
        return self.ae.get_selected_feature_indexes()

    def update_temperature(self):
        self.ae.update_temperature()