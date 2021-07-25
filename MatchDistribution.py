"""
This script derive from ToyProblem.py, and is to solve the 
distribution problem between input data to generated quantizing weight.

The quantizing weight is generated from input data through non-trained encoder.
I'm using "torch.randperm(data_length)[:weight_length]" to sample encoding data.
"""

#%% importing libs
import torch
import pytorch_lightning as pl
import torch.nn as nn
import numpy as np
from torch.utils import data as DataUtil
from torchsummaryX import summary

#%% Defining datasets
num_data = 1024
data_dim = 128
batch_size = 1024
out_dim = 4
num_quantize = num_data//4
class my_dataset(DataUtil.Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.linspace(-1,1,num_data).unsqueeze(1).repeat(1,data_dim)
        print(self.data.shape)
        print(self.data[:10])
    
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index],

data_set = my_dataset()
data_loader = DataUtil.DataLoader(data_set,batch_size=batch_size,shuffle=False)

#%% defining Quantizing layer
class Quantizing(nn.Module):
    def __init__(self,num_quantizing:int, quantizing_dim:int,_weight:torch.Tensor=None,device=None,dtype=None):
        super().__init__()
        assert num_quantize > 0
        assert quantizing_dim > 0
        self.num_quantizing = num_quantizing
        self.quantizing_dim = quantizing_dim

        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty((num_quantizing,quantizing_dim),device=device,dtype=dtype)
            )
            nn.init.normal_(self.weight,mean=0.0,std=0.001)
        else:
            assert _weight.size(1) == quantizing_dim
            assert _weight.size(0) == num_quantizing
            self.weight = nn.Parameter(_weight.to(device).to(dtype))
        self.num_quantizing = num_quantizing
        self.quantizing_dim = quantizing_dim

    def forward(self,x:torch.Tensor):
        """
        x   : shape is (*, E), and weight shape is (Q, E). 
        return -> ( quantized : shape is (*, E), quantized_idx : shape is (*,) )
        """
        input_size = x.shape
        x = x.reshape(-1,self.quantizing_dim) # shape is (B,E)
        delta = self.weight.squeeze(0) - x.unsqueeze(1) # shape is (B, Q, E)
        dist = torch.sum(delta*delta,dim=-1) # shape is (B, Q)
        q_idx = torch.argmin(dist,dim=-1) # shape is (B, )
        q_data = self.weight[q_idx] # shape is (B, Q)
        return q_data.view(input_size), q_idx.view(input_size[:-1])

# %% Defining AutoEncoder
class Encoder(nn.Module):
    def __init__(self,in_features:int,out_features:int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,out_features),
        )

    def forward(self,x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self,in_features:int,out_features:int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features,32),nn.ReLU(),
            nn.Linear(32,64),nn.ReLU(),
            nn.Linear(64,out_features),nn.Tanh(),
        )
    def forward(self,x):
        return self.layers(x)

class VQ_AutoEncoder(pl.LightningModule):

    def __init__(self,in_features:int, hidden_features:int, num_quantizing:int,lr:float = 0.001):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_quantizing = num_quantizing
        self.lr = lr
        # defining criterion
        self.criterion = nn.MSELoss()
        
        # defining histgram_data
        self.histogram_data = torch.zeros((num_quantizing,),dtype=torch.int)

        # layers
        self.encoder = Encoder(in_features,hidden_features)
            # set weight tensor from input data.
        encoding_data = data_set.data[torch.randperm(len(data_set))[:num_quantizing]]
        with torch.no_grad():
            weight = self.encoder(encoding_data)
        self.quantizer = Quantizing(num_quantizing,hidden_features,weight)
        self.decoder = Decoder(hidden_features,in_features)

    def forward(self,x:torch.Tensor):
        x = self.encoder(x)
        x,_ = self.quantizer(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=self.lr)

    def on_epoch_start(self) -> None:
        self.histogram_data.zero_()

    def training_step(self,batch,idx):
        data, = batch
        h = self.encoder(data)
        q,q_idx = self.quantizer(h)
        decoded = self.decoder(h) # non quantize
        inner_loss = self.criterion(q,h)
        true_loss = self.criterion(decoded,data)
        loss = inner_loss+true_loss
        self.log('loss',loss)
        self.log('inner_loss',loss)
        self.log('true_loss',true_loss)
        
        # add to histogram 
        uniq,cnt = torch.unique(q_idx,return_counts=True)
        self.histogram_data[uniq.cpu()] += cnt.cpu()
        self.logger.experiment.add_histogram('Quantized',q_idx,self.current_epoch)
        return loss

    def on_epoch_end(self) -> None:
        
        #self.logger.experiment.add_histogram('Quantized',self.histogram_data,self.current_epoch)
        pass

model = VQ_AutoEncoder(data_dim,out_dim,num_quantize,lr=0.001)
dummy = torch.randn(1,data_dim)
summary(model,dummy)

# %% Training
EPOCHS = 1000
trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS)
trainer.fit(model,data_loader)

#%%
model.histogram_data

#%%
w_mean = model.quantizer.weight.mean()
w_std = model.quantizer.weight.std()
d_mean,d_std = data_set.data.mean(), data_set.data.std()
print(w_mean,w_std)
print(d_mean,d_std)