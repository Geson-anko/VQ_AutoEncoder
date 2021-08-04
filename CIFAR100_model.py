""" description
<purpose>
    This is the experiment to see if the quantizing layer can be adapted to real world datasets.

<dataset>
    CIFAR100 dataset, only images.

<approch>
    I stacked down samping layer and 3 Resblocks alternately in Encoder. Decoder is inversed.
"""

import torch
import torch.nn as nn 
from torchsummaryX import summary
from hparams import CIFAR100_default as hparams
from addtional_layers import ConvNorm2d,ResBlocks2d,ConvTransposeNorm2d
from quantizing_layers import Quantizing
import pytorch_lightning as pl
from torchvision.utils import make_grid

class Encoder(nn.Module):
    channels,width,height = 3,32,32
    def __init__(self,hparams:hparams.encoder_hparams):
        super().__init__()
        self.input_size = (1,self.channels,self.width,self.height)
        self.output_size = (1,hparams.quantizing_dim)
        self.hparams = hparams
        self.model_name = hparams.model_name

        # layers
        layers = []
        _ches = [self.channels, *hparams.encoder_channels]
        for i in range(1,len(_ches)):
            layers += [
                ConvNorm2d(_ches[i-1], _ches[i], 2,2),
                nn.ReLU(),
                ResBlocks2d(_ches[i],_ches[i],hparams.res_kernel,hparams.res_num_layers)
            ]

        layers.append(nn.Flatten(1))
        fc_ch = _ches[-1] * (self.width//(2**len(hparams.encoder_channels)))**2
        fc = nn.Linear(fc_ch, hparams.quantizing_dim)
        layers.append(fc)
        self.layers = nn.Sequential(*layers)
        

    def forward(self,x:torch.Tensor):
        y = self.layers(x)
        return y

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

class Decoder(nn.Module):
    channels,width,height = 3,32,32

    def __init__(self,hparams:hparams.decoder_hparams):
        super().__init__()
        self.input_size = (1,hparams.quantizing_dim)
        self.output_size =(1,self.channels,self.width,self.height)
        self.hparams = hparams
        self.model_name = hparams.model_name

        # layers
        layers = []

        _ches = [hparams.encoder_channels[0], *[i*2 for i in hparams.encoder_channels]][::-1]
        out_width = (self.width//(2**len(hparams.encoder_channels)))
        fc_ch = _ches[0] * out_width **2
        self.fc = nn.Linear(hparams.quantizing_dim,fc_ch)
        self.__resh = (-1,_ches[0],out_width,out_width)
        
        for i in range(1,len(_ches)):
            layers += [
                ConvTransposeNorm2d(_ches[i-1],_ches[i],2,2),
                nn.ReLU(),
                ResBlocks2d(_ches[i],_ches[i],hparams.res_kernel,hparams.res_num_layers),
            ]
        self.layers = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(_ches[-1],self.channels,3,padding=1)

    def forward(self,x:torch.Tensor):
        h = torch.relu(self.fc(x))
        h = h.view(self.__resh)
        h = self.layers(h)
        y = self.out_conv(h).sigmoid()
        return y

    def summary(self):
        dummy = torch.randn(self.input_size)
        summary(self,dummy)
        
class VQ_AutoEncoder(pl.LightningModule):

    def __init__(self,hparams:hparams):
        super().__init__()
        self.model_name  = hparams.model_name
        self.my_hparams = hparams
        self.num_quantizing = hparams.num_quantizing
        self.quantizing_dim = hparams.quantizing_dim
        self.lr = hparams.lr

        # set crterion
        self.MSE = nn.MSELoss()
        
        # set histgrams
        self.q_hist = torch.IntTensor()

        # set layers
        self.encoder = Encoder(hparams.encoder_hparams)
        self.quantizer = Quantizing(hparams.num_quantizing,hparams.quantizing_dim)
        self.decoder = Decoder(hparams.decoder_hparams)

        self.input_size = self.encoder.input_size
        self.output_size = self.input_size

    def forward(self,x:torch.Tensor):
        h = self.encoder(x)
        #h,idx = self.quantizer(h)
        y = self.decoder(h)
        return y

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),self.lr)
        return optim
    
    def training_step(self,batch,idx):
        data,_ = batch
        self.data = data
        encoded = self.encoder(data)
        q_data,q_idx = self.quantizer(encoded)
        decoded = self.decoder(encoded)

        # loss
        t_loss = self.MSE(decoded,data)
        q_loss = self.MSE(q_data,encoded)
        loss = t_loss + q_loss
        self.log('target loss', t_loss)
        self.log('quantizing loss',q_loss)

        self.q_hist = torch.cat([self.q_hist,q_idx.cpu()])
        
        return loss

    @torch.no_grad()
    def on_epoch_end(self) -> None:
        if (self.current_epoch+1) % self.my_hparams.view_interval == 0:
            data = self.data[:self.my_hparams.max_view_imgs].float()
            encoded = self.encoder(data)
            q_data,q_idx = self.quantizer(encoded)
            decoded =self.decoder(encoded)
            q_decoded = self.decoder(q_data)

            grid_img = make_grid(torch.cat([decoded,data],dim=0))
            q_grid_img = make_grid(q_decoded)
            self.logger.experiment.add_image('decoded imgs',grid_img,self.current_epoch)
            self.logger.experiment.add_image('quantized imgs',q_grid_img,self.current_epoch)

            self.logger.experiment.add_histogram('Quantized Histogram',self.q_hist,self.current_epoch)

        self.q_hist = torch.IntTensor()

    def summary(self,tensorboard=False):
        from torch.utils.tensorboard import SummaryWriter
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

        if tensorboard:
            writer = SummaryWriter(comment=self.model_name)
            writer.add_graph(self,dummy)
            writer.close()

if __name__ == '__main__':
    model = VQ_AutoEncoder(hparams)
    model.summary(True)
        

