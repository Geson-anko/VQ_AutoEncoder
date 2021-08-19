model_name = 'MNIST_default'
max_view_imgs = 16
view_interval = 10

lr:float = 0.001

num_quantizing:int = 32
quantizing_dim:int = 32

def get() -> dict:
    hparam:dict = {
        "model_name":model_name,
        "num_quantizing":num_quantizing,
        "quantizing_dim":quantizing_dim,
        "lr":lr,
    }
    return hparam

class encoder_hparam:
    model_name = model_name + '_encoder'
    quantizing_dim = quantizing_dim

    def get(self) -> dict:
        hparam = {
            "model_name" : model_name,
            "quantizing_dim":quantizing_dim,
        }
        return hparam
class decoder_hparam:
    model_name = model_name + '_decoder'
    quantizing_dim = quantizing_dim

    def get(self) -> dict:
        hparam = {
            "model_name" : model_name,
            "quantizing_dim":quantizing_dim,
        }
        return hparam


