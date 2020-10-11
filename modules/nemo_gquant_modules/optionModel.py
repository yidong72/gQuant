import cupy
import numpy as np
import math
import time
import numba
from numba import cuda
from numba import njit
from numba import prange
import cudf
import torch
import torch as t
import torch.nn as nn
import torch.utils.data as t_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import nemo
from nemo.core.neural_types import NeuralType, ChannelType, LossType, LabelsType
from nemo.utils.decorators import add_port_docs

from gquant.dataframe_flow import Node
from .nemoBaseNode import NeMoBase
import nemo

import torch
from torch.utils.dlpack import from_dlpack


import cupy
cupy_batched_barrier_option = cupy.RawKernel(r'''
extern "C" __global__ void batched_barrier_option(
    float *d_s,
    float *d_d,
    const float * T,
    const float * K,
    const float * B,
    const float * S0,
    const float * sigma,
    const float * mu,
    const float * r,
    const float * d_normals,
    const long *N_STEPS,
    const long Y_STEPS,
    const long N_PATHS,
    const long N_BATCH)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned tid = threadIdx.x;
  double d_theta[6];
  double d_a[6];

  for (unsigned i = idx; i<N_PATHS * N_BATCH; i+=stride)
  {
    d_theta[0] = 0; // T
    d_theta[1] = 0; // K
    d_theta[2] = 1.0; // S_0
    d_theta[3] = 0; // mu
    d_theta[4] = 0; // sigma
    d_theta[5] = 0; // r
    for (unsigned k = 0; k < 6; k++){
      d_a[k] = 0.0;
    }
    
    int batch_id = i / N_PATHS;
    int path_id = i % N_PATHS;
    float s_curr = S0[batch_id];
    float tmp1 = mu[batch_id]/Y_STEPS;
    float tmp2 = exp(-r[batch_id]*T[batch_id]);
    float tmp3 = sqrt(1.0/Y_STEPS);
    unsigned n=0;
    double running_average = 0.0;
    for(unsigned n = 0; n < N_STEPS[batch_id]; n++){
        if (n == N_STEPS[batch_id] - 1) {
            float delta_t = T[batch_id] - n/Y_STEPS;
            tmp1 = delta_t * mu[batch_id];
            tmp3 = sqrt(abs(delta_t));
        }
        float normal = d_normals[path_id + batch_id * N_PATHS + n * N_PATHS * N_BATCH];
        
            
        // start to compute the gradient
        float factor = (1.0+tmp1+sigma[batch_id]*tmp3*normal);
        for (unsigned k=0; k < 6; k++) {
            d_theta[k] *= factor;
        }
        
        if (n == N_STEPS[batch_id] - 1){
                d_theta[0] += (mu[batch_id] + 0.5 * sigma[batch_id] * normal / tmp3) * s_curr;
                d_theta[3] += (T[batch_id] - n/Y_STEPS) * s_curr;
                d_theta[4] += tmp3 * normal * s_curr;
        }
        else {
                d_theta[3] += 1.0/Y_STEPS * s_curr;
                d_theta[4] += tmp3 * normal * s_curr;
        }
        for (unsigned k = 0; k < 6; k++) {
                d_a[k] = d_a[k]*n/(n+1.0) + d_theta[k]/(n+1.0); 
        }
        
        
        // start to compute current stock price and moving average       
       
       s_curr += tmp1 * s_curr + sigma[batch_id]*s_curr*tmp3*normal;
       running_average += (s_curr - running_average) / (n + 1.0);
       if (running_average <= B[batch_id]){
           break;
       }
    }

    float payoff = (running_average>K[batch_id] ? running_average-K[batch_id] : 0.f); 
    d_s[i] = tmp2 * payoff;
    //printf("%d, %d, %f, %f, %f, %d\n", i, idx, d_s[i], payoff, K[batch_id], batch_id);
    
    // gradient for strik 
    if (running_average > K[batch_id]){
       d_a[1] = -1.0;
       // adjust gradient for discount factor
       for (unsigned k = 0; k < 6; k++) {
            d_a[k] *= tmp2;
        }
        d_a[0] += payoff * tmp2* -r[batch_id];
        d_a[5] += payoff * tmp2* -T[batch_id];
        
    }
    else {
        for (unsigned k = 0; k < 6; k++) {
           d_a[k] = 0.0;
        }

    }
    
    for (unsigned k = 0; k < 6; k++) {
       d_d[k*N_PATHS*N_BATCH+i] = d_a[k];
    }
  }
}

''', 'batched_barrier_option')

class ParameterIter(object):
    
    def __init__(self, batch, K=200.0, S0=200.0, sigma=0.4, mu=0.2, r=0.2, T=1.9, minT=0.1, seed=None):
        self.N_BATCH = batch
        self.K = K
        self.S0 = S0
        self.sigma = sigma
        self.mu = mu
        self.r = r
        self.T = T
        self.minT = minT
        if seed is not None:
            cupy.random.seed(seed)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        """
        Parameters order (B, T, K, S0, mu, sigma, r)
        """
        X = cupy.random.rand(self.N_BATCH, 7, dtype=cupy.float32)
        # scale the [0, 1) random numbers to the correct range for each of the option parameters
        X = X * cupy.array([ 0.99, self.T, self.K, self.S0, self.mu, self.sigma, self.r], dtype=cupy.float32)
        # make sure the Barrier is smaller than the Strike price
        X[:, 0] = X[:, 0] * X[:, 2]
        X[:, 1] += self.minT 
        
        X[:, 0] += 10.0
        X[:, 2] += 10.0        
        X[:, 3] += 10.0  
        
        X[:, 4] += 0.0001
        X[:, 5] += 0.0001
        X[:, 6] += 0.0001
        return X

class SimulationIter(object):
    
    def __init__(self, para_iter, N_PATHS=102400, Y_STEPS=252):
        self.para_iter = para_iter
        self.N_PATHS = N_PATHS
        self.Y_STEPS = Y_STEPS
        self.N_BATCH = para_iter.N_BATCH
        self.block_threads = 256
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Parameters order (B, T, K, S0, mu, sigma, r)
        para = next(self.para_iter)
        B = cupy.ascontiguousarray(para[:, 0])
        T = cupy.ascontiguousarray(para[:, 1])
        K = cupy.ascontiguousarray(para[:, 2])
        S0 = cupy.ascontiguousarray(para[:, 3])
        mu = cupy.ascontiguousarray(para[:, 4])
        sigma = cupy.ascontiguousarray(para[:, 5])        
        r = cupy.ascontiguousarray(para[:, 6])

        N_STEPS = cupy.ceil(T * self.Y_STEPS).astype(cupy.int64)
        number_of_threads = self.block_threads
        number_of_blocks = (self.N_PATHS * self.N_BATCH - 1) // number_of_threads + 1
        random_elements = (N_STEPS.max()*self.N_PATHS*self.N_BATCH).item()
        randoms_gpu = cupy.random.normal(0, 1, random_elements, dtype=cupy.float32)
        output = cupy.zeros(self.N_BATCH * self.N_PATHS, dtype=cupy.float32)
        d_output = cupy.zeros(self.N_BATCH*self.N_PATHS*6, dtype=cupy.float32)
        cupy_batched_barrier_option((number_of_blocks,), (number_of_threads,),
                                    (output, d_output, T, K, B, S0, sigma, mu, r,
                                    randoms_gpu, N_STEPS, self.Y_STEPS, self.N_PATHS, self.N_BATCH))
        v = output.reshape(self.N_BATCH, self.N_PATHS).mean(axis=1)[:,None]
        b = d_output.reshape(6, self.N_BATCH, self.N_PATHS).mean(axis=2).T
        y = cupy.concatenate([v, b], axis=1)
        return para, y

# import torch
# from torch.utils.dlpack import from_dlpack
# p_iter = ParameterIter(1, seed=5)
# sim_iter = SimulationIter(p_iter)
# next(sim_iter)
# X, Y = next(sim_iter)
# X_t, Y_t = (from_dlpack(X[0].toDlpack()), from_dlpack(Y[0].toDlpack()))
# print(X_t, Y_t)
# class OptionDataSet(torch.utils.data.IterableDataset):
#     
#     def __init__(self, seed=2,  N_PATHS=102400, Y_STEPS=252, max_len=10):
#         p_iter = ParameterIter(1, seed=seed)
#         self.sim_iter = SimulationIter(p_iter, N_PATHS=N_PATHS, Y_STEPS=Y_STEPS)
#         self.num = 0
#         self.max_length = max_len
#         
#     def __len__(self):
#         return self.max_length
#         
#     def __iter__(self):
#         self.num = 0
#         return self
#     
#     def __next__(self):
#         if self.num > self.max_length:
#             raise StopIteration
#         X, Y = next(self.sim_iter)
#         return (from_dlpack(X[0].toDlpack()), from_dlpack(Y[0].toDlpack()))
class OptionDataSet(torch.utils.data.Dataset):
    def __init__(self,  seed=2,  N_PATHS=102400, Y_STEPS=252, max_len=10):
        p_iter = ParameterIter(1, seed=seed)
        self.sim_iter = SimulationIter(p_iter, N_PATHS=N_PATHS, Y_STEPS=Y_STEPS)
        self.num = 0
        self.max_length = max_len

    def __getitem__(self, index):
        X, Y = next(self.sim_iter)
        return (from_dlpack(X[0].toDlpack()), from_dlpack(Y[0].toDlpack()))        

    def __len__(self):
        return self.max_length    

class OptionDataLayer(nemo.backends.pytorch.nm.DataLayerNM):
    
    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports
        """
        return {
            "x": NeuralType(('B', 'D'), ChannelType()),
            "y": NeuralType(('B', 'D'), LabelsType()),
    }
    

    def __init__(self, seed=2,  N_PATHS=102400, Y_STEPS=252, max_len=10, batch_size=2, name=None):
        super().__init__(name=name)
        self._batch_size = batch_size
        self._dataset = OptionDataSet(seed, N_PATHS, Y_STEPS, max_len)
        self._data_iterator = t_utils.DataLoader(self._dataset, batch_size=self._batch_size)

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._data_iterator

class MSELoss(nemo.backends.pytorch.nm.LossNM):
    @property
    @add_port_docs()
    def input_ports(self):
        return {
            "predictions": NeuralType(('B', 'D'), ChannelType()),
            "target": NeuralType(('B', 'D'), LabelsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, name=None):
        super().__init__(name=name)
        self._criterion = nn.MSELoss()

    def _loss_function(self, **kwargs):
        # print('loss', self._criterion(*(kwargs.values())))
        return self._criterion(*(kwargs.values()))


class NetLayer(nemo.backends.pytorch.nm.TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        Returns:
          A (dict) of module's input ports names to NeuralTypes mapping
        """
        return {"x": NeuralType(('B', 'D'), ChannelType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        Returns:
          A (dict) of module's output ports names to NeuralTypes mapping
        """
        return {"y_pred": NeuralType(('B', 'D'), ChannelType())}


    def __init__(self, hidden=512, name=None):
        super().__init__(name=name)
        self.fc1 = nn.Linear(7, hidden).cuda()
        self.fc2 = nn.Linear(hidden, hidden).cuda()
        self.fc3 = nn.Linear(hidden, hidden).cuda()
        self.fc4 = nn.Linear(hidden, hidden).cuda()
        self.fc5 = nn.Linear(hidden, hidden).cuda()
        self.fc6 = nn.Linear(hidden, 1).cuda()
        self.register_buffer('norm',
                             torch.tensor([198.0,
                                           2.0,
                                           200.0,
                                           200.0,
                                           0.2,
                                           0.4,
                                           0.2]).cuda())

    def _forward(self, x):
        x = x / self.norm
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = F.elu(self.fc5(x))
        y = self.fc6(x)
        return y
        
    def forward(self, x):
        """
        Parameters order (B, T, K, S0, mu, sigma, r)
        """
        y = self._forward(x)
        # inputs = x.clone().detach()
        # inputs = x
        x.requires_grad = True
        # instead of using loss.backward(), use torch.autograd.grad() to compute gradients
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        loss_grads = grad(self._forward(x).cuda().sum(), x, create_graph=True)
        return torch.cat((y, loss_grads[0][:, 1:]), axis=1)

class OptionDataLayerNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, OptionDataLayer)



class OptionPriceNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, NetLayer)



class OptionMSELossNode(NeMoBase, Node):
    def init(self):
        NeMoBase.init(self, MSELoss)

# _dataset = OptionDataSet(seed=2,  N_PATHS=102400, Y_STEPS=252, max_len=10)
# print(len(_dataset))
# _data_iterator = t_utils.DataLoader(_dataset, batch_size=2)
# for i in _data_iterator:
#     print(i[0].shape, i[1].shape)
