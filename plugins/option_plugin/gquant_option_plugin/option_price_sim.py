import cupy

cupy_barrier_option = cupy.RawKernel(r'''
extern "C" __global__ void batched_barrier_option(
    float *d_s,
    float *d_d,
    const float K,
    const float B,
    const float S0,
    const float sigma,
    const float mu,
    const float r,
    const float * d_normals,
    const long N_STEPS,
    const long Y_STEPS,
    const long N_PATHS)
{
  unsigned idx =  threadIdx.x + blockIdx.x * blockDim.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned tid = threadIdx.x;
  double d_theta[5];
  double d_a[5];

  for (unsigned i = idx; i<N_PATHS; i+=stride)
  {
    d_theta[0] = 0; // K
    d_theta[1] = 1.0; // S_0
    d_theta[2] = 0; // mu
    d_theta[3] = 0; // sigma
    d_theta[4] = 0; // r
    for (unsigned k = 0; k < 5; k++){
      d_a[k] = 0.0;
    }
    
    int path_id = i;
    float s_curr = S0;
    float tmp1 = mu/Y_STEPS;
    float tmp2 = exp(-r);
    float tmp3 = sqrt(1.0/Y_STEPS);
    unsigned n=0;
    double running_average = 0.0;
    for(unsigned n = 0; n < N_STEPS; n++){

        float normal = d_normals[path_id + n * N_PATHS];
                
        // start to compute the gradient
        float factor = (1.0+tmp1+sigma*tmp3*normal);
        for (unsigned k=0; k < 5; k++) {
            d_theta[k] *= factor;
        }
        

        d_theta[2] += 1.0/Y_STEPS * s_curr;
        d_theta[3] += tmp3 * normal * s_curr;

        for (unsigned k = 0; k < 5; k++) {
                d_a[k] = d_a[k]*n/(n+1.0) + d_theta[k]/(n+1.0); 
        }
        
        
        // start to compute current stock price and moving average       
       
       s_curr += tmp1 * s_curr + sigma*s_curr*tmp3*normal;
       running_average += (s_curr - running_average) / (n + 1.0);
       if (running_average <= B){
           break;
       }
    }

    float payoff = (running_average>K ? running_average-K : 0.f); 
    d_s[i] = tmp2 * payoff;
    
    // gradient for strik 
    if (running_average > K){
       d_a[0] = -1.0;
       // adjust gradient for discount factor
       for (unsigned k = 0; k < 5; k++) {
            d_a[k] *= tmp2;
        }
        d_a[4] += - payoff * tmp2;
        
    }
    else {
        for (unsigned k = 0; k < 5; k++) {
           d_a[k] = 0.0;
        }

    }
    
    for (unsigned k = 0; k < 5; k++) {
       d_d[k*N_PATHS+i] = d_a[k];
    }
  }
}
''', 'batched_barrier_option')


class ParameterIter(object):

    def __init__(self, K=200.0, S0=200.0, sigma=0.4,
                 mu=0.2, r=0.2, seed=None):
        self.K = K
        self.S0 = S0
        self.sigma = sigma
        self.mu = mu
        self.r = r
        if seed is not None:
            cupy.random.seed(seed)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Parameters order (B, K, S0, mu, sigma, r)
        """
        X = cupy.random.rand(6, dtype=cupy.float32)
        # scale the [0, 1) random numbers to the correct range for
        # each of the option parameters
        X = X * cupy.array([0.99, self.K, self.S0, self.mu,
                            self.sigma, self.r], dtype=cupy.float32)
        # make sure the Barrier is smaller than the Strike price
        X[0] = X[0] * X[1]
        X[0] += 10.0
        X[1] += 10.0
        X[2] += 10.0
        X[3] += 0.0001
        X[4] += 0.0001
        X[5] += 0.0001
        return X


class SimulationIter(object):

    def __init__(self, para_iter, N_PATHS=102400, Y_STEPS=365):
        self.para_iter = para_iter
        self.N_PATHS = N_PATHS
        self.Y_STEPS = Y_STEPS
        self.block_threads = 256

    def __iter__(self):
        return self

    def __next__(self):
        # Parameters order (B, K, S0, mu, sigma, r)
        para = next(self.para_iter)
        B = para[0].item()
        K = para[1].item()
        S0 = para[2].item()
        mu = para[3].item()
        sigma = para[4].item()
        r = para[5].item()

        N_STEPS = self.Y_STEPS
        number_of_threads = self.block_threads
        number_of_blocks = (self.N_PATHS
                            - 1) // number_of_threads + 1
        random_elements = int(N_STEPS * self.N_PATHS)
        randoms_gpu = cupy.random.normal(0, 1, random_elements,
                                         dtype=cupy.float32)
        output = cupy.zeros(self.N_PATHS, dtype=cupy.float32)
        d_output = cupy.zeros(self.N_PATHS*5, dtype=cupy.float32)
        cupy_barrier_option((number_of_blocks,), (number_of_threads,),
                            (output, d_output,
                             cupy.float32(K),
                             cupy.float32(B),
                             cupy.float32(S0),
                             cupy.float32(sigma),
                             cupy.float32(mu),
                             cupy.float32(r),
                             randoms_gpu, N_STEPS, self.Y_STEPS,
                             self.N_PATHS))
        v = output.mean()
        b = d_output.reshape(5, self.N_PATHS).mean(axis=1)
        # gradient
        # dt, dK, dS, dmu, dsigma, dr
        y = cupy.concatenate([cupy.array([v]), b])
        return para, y
