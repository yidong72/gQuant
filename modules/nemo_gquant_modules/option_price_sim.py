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
        float normal = d_normals[path_id + batch_id * N_PATHS
                                 + n * N_PATHS * N_BATCH];


        // start to compute the gradient
        float factor = (1.0+tmp1+sigma[batch_id]*tmp3*normal);
        for (unsigned k=0; k < 6; k++) {
            d_theta[k] *= factor;
        }

        if (n == N_STEPS[batch_id] - 1){
                d_theta[0] += (mu[batch_id] +
                    0.5 * sigma[batch_id] * normal / tmp3) * s_curr;
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

    float payoff = (running_average>K[batch_id] ? running_average-K[batch_id]
                    : 0.f);
    d_s[i] = tmp2 * payoff;

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

    def __init__(self, batch, K=200.0, S0=200.0, sigma=0.4,
                 mu=0.2, r=0.2, T=1.9, minT=0.1, seed=None):
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
        # scale the [0, 1) random numbers to the correct range for
        # each of the option parameters
        X = X * cupy.array([0.99, self.T, self.K, self.S0, self.mu,
                            self.sigma, self.r], dtype=cupy.float32)
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
        number_of_blocks = (self.N_PATHS * self.N_BATCH
                            - 1) // number_of_threads + 1
        random_elements = (N_STEPS.max()*self.N_PATHS*self.N_BATCH).item()
        randoms_gpu = cupy.random.normal(0, 1, random_elements,
                                         dtype=cupy.float32)
        output = cupy.zeros(self.N_BATCH * self.N_PATHS, dtype=cupy.float32)
        d_output = cupy.zeros(self.N_BATCH*self.N_PATHS*6, dtype=cupy.float32)
        cupy_batched_barrier_option((number_of_blocks,), (number_of_threads,),
                                    (output, d_output, T, K, B, S0, sigma, mu,
                                     r, randoms_gpu, N_STEPS, self.Y_STEPS,
                                     self.N_PATHS, self.N_BATCH))
        v = output.reshape(self.N_BATCH,
                           self.N_PATHS).mean(axis=1)[:, None]
        b = d_output.reshape(6, self.N_BATCH, self.N_PATHS).mean(axis=2).T
        y = cupy.concatenate([v, b], axis=1)
        return para, y
