import torch
import torch.nn as nn
from typing import Union,Tuple

class Quantizing(nn.Module):
    """
    This is quantizing layer.
    """
    __initialized:bool = True

    def __init__(
        self, num_quantizing:int, quantizing_dim:int, _weight:torch.Tensor = None,
        initialize_by_dataset:bool = True, mean:float = 0.0, std:float = 1.0,
        dtype:torch.dtype = None, device:torch.device = None
        ):
        super().__init__()
        assert num_quantizing > 0
        assert quantizing_dim > 0
        self.num_quantizing = num_quantizing
        self.quantizing_dim = quantizing_dim
        self.initialize_by_dataset = initialize_by_dataset
        self.mean,self.std = mean,std

        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty(num_quantizing, quantizing_dim ,dtype=dtype,device=device)
            )
            nn.init.normal_(self.weight, mean=mean, std=std)

            if initialize_by_dataset:
                self.__initialized = False
                self.__initialized_length = 0
        
        else:
            assert _weight.dim() == 2
            assert _weight.size(0) == num_quantizing
            assert _weight.size(1) == quantizing_dim
            self.weight = nn.Parameter(_weight.to(device).to(dtype))

        self.register_buffer('quantizing',self.weight)

    def forward(self,x:torch.Tensor) -> Tuple[torch.Tensor]:
        """
        x   : shape is (*, E), and weight shape is (Q, E). 
        return -> ( quantized : shape is (*, E), quantized_idx : shape is (*,) )
        """
        input_size = x.shape
        h = x.view(-1,self.quantizing_dim) # shape is (B,E)

        if not self.__initialized and self.initialize_by_dataset:
            getting_len = self.num_quantizing - self.__initialized_length
            init_weight = h[torch.randperm(len(h))[:getting_len]]
            
            _until = self.__initialized_length + init_weight.size(0)
            self.weight.data[self.__initialized_length:_until] = init_weight
            self.__initialized_length = _until
            #print('replaced weight')

            if _until >= self.num_quantizing:
                self.__initialized = True
                #print('initialized')
        
        delta = self.weight.unsqueeze(0) - h.unsqueeze(1) # shape is (B, Q, E)
        dist =torch.sum(delta*delta, dim=-1) # shape is (B, Q)
        q_idx = torch.argmin(dist,dim=-1) # shape is (B,)
        q_data = self.weight[q_idx] # shape is (B, E)

        return q_data.view(input_size), q_idx.view(input_size[:1])

    def from_idx(self,idx:torch.Tensor) -> torch.Tensor:
        """
        idx: shape is (*, ). int tensor.
        return -> (*, E) float tensor
        """
        input_size = idx.shape
        i = idx.view(-1)
        q_data = self.weight[i].view(*input_size,self.quantizing_dim)
        return q_data

    def load_state_dict(self, state_dict, strict: bool):
        self.__initialized = True
        return super().load_state_dict(state_dict, strict=strict)
    
    def __repr__(self):
        s = f'Quantizing({self.num_quantizing}, {self.quantizing_dim})'
        return s

class Quantizing_cossim(nn.Module):
    """
    This is quantizing layer.
    """
    __initialized:bool = True

    def __init__(
        self, num_quantizing:int, quantizing_dim:int, _weight:torch.Tensor = None,
        initialize_by_dataset:bool = True, mean:float = 0.0, std:float = 1.0, eps:float = 1e-8,
        dtype:torch.dtype = None, device:torch.device = None
        ):
        super().__init__()
        assert num_quantizing > 0
        assert quantizing_dim > 0
        self.num_quantizing = num_quantizing
        self.quantizing_dim = quantizing_dim
        self.initialize_by_dataset = initialize_by_dataset
        self.mean,self.std = mean,std
        self.eps = eps

        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty(num_quantizing, quantizing_dim ,dtype=dtype,device=device)
            )
            nn.init.normal_(self.weight, mean=mean, std=std)

            if initialize_by_dataset:
                self.__initialized = False
                self.__initialized_length = 0
        
        else:
            assert _weight.dim() == 2
            assert _weight.size(0) == num_quantizing
            assert _weight.size(1) == quantizing_dim
            self.weight = nn.Parameter(_weight.to(device).to(dtype))

        self.register_buffer('quantizing',self.weight)

    def forward(self,x:torch.Tensor) -> Tuple[torch.Tensor]:
        """
        x   : shape is (*, E), and weight shape is (Q, E). 
        return -> ( quantized : shape is (*, E), quantized_idx : shape is (*,) )
        """
        input_size = x.shape
        h = x.view(-1,self.quantizing_dim) # shape is (B,E)

        if not self.__initialized and self.initialize_by_dataset:
            getting_len = self.num_quantizing - self.__initialized_length
            init_weight = h[torch.randperm(len(h))[:getting_len]]
            
            _until = self.__initialized_length + init_weight.size(0)
            self.weight.data[self.__initialized_length:_until] = init_weight
            self.__initialized_length = _until
            #print('replaced weight')

            if _until >= self.num_quantizing:
                self.__initialized = True
                #print('initialized')
        
        dist =self.calculate_distance(h) # shape is (B, Q)
        q_idx = torch.argmin(dist,dim=-1) # shape is (B,)
        q_data = self.weight[q_idx] # shape is (B, E)

        return q_data.view(input_size), q_idx.view(input_size[:1])

    def from_idx(self,idx:torch.Tensor) -> torch.Tensor:
        """
        idx: shape is (*, ). int tensor.
        return -> (*, E) float tensor
        """
        input_size = idx.shape
        i = idx.view(-1)
        q_data = self.weight[i].view(*input_size,self.quantizing_dim)
        return q_data

    def load_state_dict(self, state_dict, strict: bool):
        self.__initialized = True
        return super().load_state_dict(state_dict, strict=strict)
    
    def __repr__(self):
        s = f'Quantizing({self.num_quantizing}, {self.quantizing_dim})'
        return s

    def calculate_distance(self,x:torch.Tensor) -> torch.Tensor:
        """
        x: shape is (B, *), float tensor
        """
        assert x.dim() == 2
        dot = torch.matmul(x,self.weight.T)
        x_l2n = torch.linalg.norm(x,dim=-1)[:,None]
        w_l2n = torch.linalg.norm(self.weight,dim=-1)[None,:]
        norm = torch.matmul(x_l2n,w_l2n)
        norm[norm<self.eps] = self.eps
        cos_sim = dot / norm
        return -cos_sim + 1



if __name__ == '__main__':
    l = Quantizing_cossim(4,3,torch.randn(4,3))
    l(torch.randn(4,3))