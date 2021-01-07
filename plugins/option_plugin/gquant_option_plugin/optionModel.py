import torch
# import torch.nn as nn
# import torch.utils.data as t_utils
# import torch.nn.functional as F
# from torch.autograd import grad
# import inspect
# import nemo
# from nemo.core.neural_types import (NeuralType, ChannelType,
#                                     LossType, LabelsType)
# from nemo.utils.decorators import add_port_docs

from gquant.dataframe_flow import Node, PortsSpecSchema, ConfSchema, NodePorts, MetaData
# from .nemoBaseNode import NeMoBase
from .option_price_sim import ParameterIter, SimulationIter
from torch.utils.dlpack import from_dlpack


class OptionDataSet(torch.utils.data.Dataset):
    def __init__(self,  sim_iter, max_len=10):
        self.sim_iter = sim_iter
        self.num = 0
        self.max_length = max_len

    def __getitem__(self, index):
        X, Y = next(self.sim_iter)
        return (from_dlpack(X[0].toDlpack()), from_dlpack(Y[0].toDlpack()))

    def __len__(self):
        return self.max_length


# class OptionDataLayer(nemo.backends.pytorch.nm.DataLayerNM):
# 
#     @property
#     @add_port_docs()
#     def output_ports(self):
#         """Returns definitions of module output ports
#         """
#         return {
#             "x": NeuralType(('B', 'D'), ChannelType()),
#             "y": NeuralType(('B', 'D'), LabelsType())
#             }
# 
#     def __init__(self, sim_iter, max_len=100, batch_size=16, name=None):
#         super().__init__(name=name)
#         self._batch_size = batch_size
#         self._dataset = OptionDataSet(sim_iter, max_len)
#         self._data_iterator = t_utils.DataLoader(self._dataset,
#                                                  batch_size=self._batch_size)
# 
#     def __len__(self):
#         return len(self._dataset)
# 
#     @property
#     def dataset(self):
#         return None
# 
#     @property
#     def data_iterator(self):
#         return self._data_iterator
# 
# 
# class MSELoss(nemo.backends.pytorch.nm.LossNM):
#     @property
#     @add_port_docs()
#     def input_ports(self):
#         return {
#             "predictions": NeuralType(('B', 'D'), ChannelType()),
#             "target": NeuralType(('B', 'D'), LabelsType()),
#         }
# 
#     @property
#     @add_port_docs()
#     def output_ports(self):
#         """Returns definitions of module output ports.
#         """
#         return {"loss": NeuralType(elements_type=LossType())}
# 
#     def __init__(self, name=None):
#         super().__init__(name=name)
#         self._criterion = nn.MSELoss()
# 
#     def _loss_function(self, **kwargs):
#         # print('loss', self._criterion(*(kwargs.values())))
#         return self._criterion(*(kwargs.values()))
# 
# 
# class NetLayer(nemo.backends.pytorch.nm.TrainableNM):
#     @property
#     @add_port_docs()
#     def input_ports(self):
#         """Returns definitions of module input ports.
#         Returns:
#           A (dict) of module's input ports names to NeuralTypes mapping
#         """
#         return {"x": NeuralType(('B', 'D'), ChannelType())}
# 
#     @property
#     @add_port_docs()
#     def output_ports(self):
#         """Returns definitions of module output ports.
#         Returns:
#           A (dict) of module's output ports names to NeuralTypes mapping
#         """
#         return {"y_pred": NeuralType(('B', 'D'), ChannelType())}
# 
#     def __init__(self, hidden=512, layers=4, name=None):
#         super().__init__(name=name)
#         self.network = nn.ModuleList()
#         for k in range(layers - 1):
#             if k == 0:
#                 self.network.append(nn.Linear(7, hidden))
#             else:
#                 self.network.append(nn.Linear(hidden, hidden))
#         self.last = nn.Linear(hidden, 1).cuda()
#         self.network = self.network.cuda()
#         self.register_buffer('norm',
#                              torch.tensor([198.0,
#                                            2.0,
#                                            200.0,
#                                            200.0,
#                                            0.2,
#                                            0.4,
#                                            0.2]).cuda())
# 
#     def _forward(self, x):
#         x = x / self.norm
#         for _, l in enumerate(self.network):
#             x = F.elu(l(x))
#         y = self.last(x)
#         return y
# 
#     def forward(self, x):
#         """
#         Parameters order (B, T, K, S0, mu, sigma, r)
#         """
#         y = self._forward(x)
#         # inputs = x.clone().detach()
#         # inputs = x
#         with torch.set_grad_enabled(True):
#             x.requires_grad = True
#             # instead of using loss.backward(), use torch.autograd.grad()
#             # to compute gradients
#             # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
#             loss_grads = grad(self._forward(x).cuda().sum(), x,
#                               create_graph=True)
#         return torch.cat((y, loss_grads[0][:, 1:]), axis=1)


class ParaNode(Node):

    def ports_setup(self):
        input_ports = {}
        output_ports = {
            'para_out': {
                PortsSpecSchema.port_type: ParameterIter
            },
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        json = {
            "title": "Source node configure",
            "type": "object",
            "properties": {
                "seed": {
                    "type": ["integer", "null"],
                    "title": "seed number",
                    "description": "seed number for random numbers",
                    "default": None
                }
            }
        }
        ui = {}
        return ConfSchema(json=json, ui=ui)

    def init(self):
        pass

    def meta_setup(self):
        """
        Parameters order (B, T, K, S0, mu, sigma, r)
        """
        required = {}
        columns_out = {
            'para_out': {
                'B': 'float32',
                "T": 'float32',
                'K': 'float32',
                "S0": "float32",
                "mu": "float32",
                "sigma": "float32",
                "r": "float32"
            },
        }
        return MetaData(inports=required, outports=columns_out)

    def process(self, inputs):
        output = {}
        it = ParameterIter(1, seed=self.conf.get('seed', None))
        output.update({'para_out': it})
        return output


class SimNode(Node):

    def ports_setup(self):
        input_ports = {
             'para_in': {
                PortsSpecSchema.port_type: ParameterIter}
        }
        output_ports = {
            'sim_out': {
                PortsSpecSchema.port_type: SimulationIter}
        }
        return NodePorts(inports=input_ports, outports=output_ports)

    def conf_schema(self):
        json = {
            "title": "Monte Carlo Sim node configure",
            "type": "object",
            "properties": {
                "N_PATHS": {
                    "type": "integer",
                    "title": "Number Paths",
                    "description": "Number of paths in the simulation",
                    "default": 102400
                },
                "Y_STEPS": {
                    "type": "integer",
                    "title": "Steps per year",
                    "description": "Number of steps for one year",
                    "default": 252
                },
            }
        }
        ui = {}
        return ConfSchema(json=json, ui=ui)

    def init(self):
        pass

    def meta_setup(self):
        required = {
            "para_in": {
                'B': 'float32',
                "T": 'float32',
                'K': 'float32',
                "S0": "float32",
                "mu": "float32",
                "sigma": "float32",
                "r": "float32"
            },
        }
        columns_out = {
            'sim_out': {
                'X': 'parameters',
                'Y': 'values'
            },
        }
        return MetaData(inports=required, outports=columns_out)

    def process(self, inputs):
        it = inputs['para_in']
        sit = SimulationIter(it, N_PATHS=self.conf.get('N_PATHS', 102400),
                             Y_STEPS=self.conf.get('Y_STEPS', 252))
        output = {}
        output.update({'sim_out': sit})
        return output


# class OptionDataLayerNode(NeMoBase, Node):
# 
#     def init(self):
#         NeMoBase.init(self, OptionDataLayer)
# 
#     def get_conf_parameters(self, class_obj):
#         conf = super().get_conf_parameters(class_obj)
#         del conf['sim_iter']
#         return conf
# 
#     def get_parameters(self, class_obj, conf, inputs):
#         init_fun = class_obj.__init__
#         sig = inspect.signature(init_fun)
#         init_para = {}
#         for key in sig.parameters.keys():
#             if key == 'sim_iter':
#                 continue
#             if key == 'self':
#                 # ignore the self
#                 continue
#             if key in conf:
#                 init_para[key] = conf[key]
#             else:
#                 pass
#         init_para['sim_iter'] = inputs['sim_iter']
#         return init_para
# 
#     def ports_setup(self):
#         port_type = PortsSpecSchema.port_type
#         ports = NeMoBase.ports_setup(self)
#         ports.inports['sim_iter'] = {port_type: SimulationIter}
#         return ports
# 
# 
# class OptionPriceNode(NeMoBase, Node):
#     def init(self):
#         NeMoBase.init(self, NetLayer)
# 
# 
# class OptionMSELossNode(NeMoBase, Node):
#     def init(self):
#         NeMoBase.init(self, MSELoss)
