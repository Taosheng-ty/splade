from splade.index import index
from splade.all import train_index_retrieve
from omegaconf import DictConfig
from omegaconf import OmegaConf
# conf = OmegaConf.create({"config" : {"checkpoint_dir": "experiments/debug/checkpoint", "index_dir": "experiments/debug/index", "out_dir": "experiments/debug/out"}})
conf = OmegaConf.load('conf/config_default.yaml')
OmegaConf.update(conf,"config",{"checkpoint_dir": "experiments/debug/checkpoint", "index_dir": "experiments/debug/index", "out_dir": "experiments/debug/out"})
train_index_retrieve(conf)
print(conf)