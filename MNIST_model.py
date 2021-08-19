import torch
import torch.nn as nn
from torchsummaryX import summary
from quantizing_layers import Quantizing,Quantizing_cossim,ConsinSimilarityLoss
import pytorch_lightning as pl
from torchvision.utils import make_grid
from hparams import MNIST_default as hparams
import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
    channels,width,height = 1,28,28
    def __init__(self,hparams:hparams.encoder_hparam):
        super().__init__()
        self.input_size = (1,self.channels,self.width,self.height)
        self.output_size = (1,hparams.quantizing_dim)
        self.hparams = hparams
        self.model_name = hparams.model_name

        self.layers = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,hparams.quantizing_dim),nn.Tanh(),
        )

    def forward(self,x:torch.Tensor):
        y = self.layers(x)
        return y
    
    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

class Decoder(nn.Module):
    channels,width,height = 1,28,28
    def __init__(self,hparams:hparams.decoder_hparam):
        super().__init__()
        self.input_size = (1,hparams.quantizing_dim)
        self.output_size = (1,self.channels,self.width,self.height)
        self.hparams = hparams
        self.model_name = hparams.model_name

        self.layers = nn.Sequential(
            nn.Linear(hparams.quantizing_dim,128),nn.ReLU(),
            nn.Linear(128,256),nn.ReLU(),
            nn.Linear(256,784),nn.Sigmoid(),
        )
    
    def forward(self,x:torch.Tensor):
        y = self.layers(x)
        y = y.view(-1,self.channels,self.height,self.width)
        return y

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

class VQ_AutoEncoder(pl.LightningModule):

    def __init__(self,hparams:hparams):
        super().__init__()
        self.model_name = hparams.model_name
        self.my_hparams = hparams
        self.num_quantizing = hparams.num_quantizing
        self.quantizing_dim = hparams.quantizing_dim
        self.lr = hparams.lr
        self.my_hparams_dict = hparams.get()

        # set criterion
        self.reconstruction_loss = nn.MSELoss()
        self.quantizing_loss = nn.MSELoss()
        
        # set histogram
        self.q_hist = torch.zeros(self.num_quantizing,dtype=torch.int)
        self.q_hist_idx = np.arange(self.num_quantizing)
        # set layers
        self.encoder = Encoder(hparams.encoder_hparam)
        self.quantizer = Quantizing(hparams.num_quantizing,hparams.quantizing_dim)
        self.decoder = Decoder(hparams.decoder_hparam)

        self.input_size = self.encoder.input_size
        self.output_size = self.input_size

    def forward(self,x:torch.Tensor):
        h = self.encoder(x)
        Qout,Qidx = self.quantizer(h)
        y = self.decoder(Qout)
        return y

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),self.lr)
        return optim
    
    @torch.no_grad()
    def set_quantizing_weight(self,data_loader,device='cpu'):
        for batch in data_loader:
            data,_ = batch
            data = data.to(device)
            Eout = self.encoder(data)
            _ = self.quantizer(Eout)
            if self.quantizer.isInitialized():
                break

        torch.cuda.empty_cache()

    def on_fit_start(self) -> None:
        self.logger.log_hyperparams(self.my_hparams_dict)
#    def on_train_start(self) -> None:
#        self.logger.experiment.add_hparams(self.my_hparams_dict,dict(),run_name=self.model_name)
#        self._set_hparams(self.my_hparams_dict)
#    def on_epoch_start(self) -> None:
#        if self.current_epoch == 0:
#            self.logger.experiment.add_hparams(self.my_hparams_dict,dict(),run_name=self.model_name)

    def training_step(self,batch,idx):
        data,_  = batch
        self.view_data = data
        Eout = self.encoder(data)
        Qtgt = Eout.detach()
        Qout,Qidx = self.quantizer(Qtgt)
        out = self.decoder(Eout)

        # loss
        r_loss = self.reconstruction_loss(out,data)
        q_loss = self.quantizing_loss(Qout,Qtgt)
        loss = r_loss + q_loss

        # log
        rq_loss = self.reconstruction_loss(self.decoder(Qout),data)
        self.log('loss',loss)
        self.log('reconstruction_loss',r_loss)
        self.log('quantizing_loss',q_loss)
        self.log('reconstructed_quantizing_loss',rq_loss)

        idx,count = torch.unique(Qidx,return_counts = True)
        self.q_hist[idx.cpu()] += count.cpu()
        return loss

    @torch.no_grad()
    def on_epoch_end(self) -> None:
        self.logger.experiment.add_hparams(self.my_hparams_dict,dict(),run_name=self.model_name)
        if (self.current_epoch+1) % self.my_hparams.view_interval ==0:
            # image logging
            data = self.view_data[:self.my_hparams.max_view_imgs].float()
            data_len = len(data)
            Eout = self.encoder(data)
            Qout,Qidx = self.quantizer(Eout)
            out = self.decoder(Eout)
            Qdecoded = self.decoder(Qout)

            grid_img = make_grid(torch.cat([data,out,Qdecoded],dim=0),nrow=data_len)
            self.logger.experiment.add_image("MNIST Quantizings",grid_img,self.current_epoch)

            # histogram logging
            fig = plt.figure()
            ax = fig.subplots()
            ax.bar(self.q_hist_idx,self.q_hist)
            
            quantized_num = len(self.q_hist[self.q_hist!=0])
            q_text = f'{quantized_num}/{self.num_quantizing}'
            ax.text(0.9,1.05,q_text,ha='center',va='center',transform=ax.transAxes,fontsize=12)
            ax.set_xlabel('weight index')
            ax.set_ylabel('num')
            self.logger.experiment.add_figure('Quantizing Histogram',fig,self.current_epoch)
            
        self.q_hist.zero_()

    def summary(self,tensorboard=False):
        from torch.utils.tensorboard import SummaryWriter
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

        if tensorboard:
            writer = SummaryWriter(comment=self.model_name)
            writer.add_graph(self,dummy)
            writer.close()

class VQ_AutoEncoder_cossim(VQ_AutoEncoder):
    def __init__(self,hparams:hparams):
        super().__init__(hparams)
        self.quantizing_loss = ConsinSimilarityLoss()
        self.quantizer = Quantizing_cossim(hparams.num_quantizing,hparams.quantizing_dim)


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader
    model = VQ_AutoEncoder(hparams)
    data_set = MNIST('data',False,download=True,
        transform= transforms.Compose(
            [transforms.ToTensor()]
    ))
    data_loader = DataLoader(data_set,hparams.num_quantizing,True)
    model.set_quantizing_weight(data_loader,'cpu')
    model.summary(False)
    