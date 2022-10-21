import sys
import bisect
import torch
import torch.nn as nn
import networkx as nx

# compute the running average
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.vals = []
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.vals = []
        self.val = None
        self.avg = 0

    def update(self, val):
        self.vals.append(val)
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# SoftPlus activation function add epsilon
class SoftPlus(nn.Module):

    def __init__(self, beta=1.0, threshold=20, epsilon=1.0e-15, dim=None):
        super(SoftPlus, self).__init__()
        self.Softplus = nn.Softplus(beta, threshold)
        self.epsilon = epsilon
        self.dim = dim

    def forward(self, x):
        # apply softplus to first dim dimension
        if self.dim is None:
            result = self.Softplus(x) + self.epsilon
        else:
            result = torch.cat((self.Softplus(x[..., :self.dim])+self.epsilon, x[..., self.dim:]), dim=-1)

        return result


# multi-layer perceptron
class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=20, num_hidden=0, activation=nn.CELU()):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.uniform_(m.bias, a=-0.1, b=0.1)

        self.activation = activation

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        return self.linears[-1](x)

# This function need to be stateless
class ODEJumpFunc(nn.Module):

    def __init__(self, dim_c, dim_h, dim_N, dim_E, dim_hidden=20, num_hidden=0, activation=nn.CELU(), ortho=False,
                 jump_type="read", evnts=[], evnt_align=False, evnt_embedding="discrete",
                 graph=None, aggregation=None):
        super(ODEJumpFunc, self).__init__()

        self.dim_c = dim_c  # internal state
        self.dim_h = dim_h  # event memory with decay
        self.dim_N = dim_N  # number of event type
        self.dim_E = dim_E  # dimension for encoding of event itself
        self.ortho = ortho  # default false
        self.evnt_embedding = evnt_embedding  # defalut discrete

        assert jump_type in ["simulate", "read"], "invalide jump_type, must be one of [simulate, read]"
        self.jump_type = jump_type
        assert (jump_type == "simulate" and len(evnts) == 0) or jump_type == "read"
        self.evnts = evnts
        self.evnt_align = evnt_align

        # F is dynamics for c(t), dc/dt = F(z)
        if graph is None:
            self.F = MLP(dim_c+dim_h, dim_c, dim_hidden, num_hidden, activation)
        
        # G is the decay effect of event memory h(t)
        self.G = nn.Sequential(MLP(dim_c, dim_h, dim_hidden, num_hidden, activation), nn.Softplus())
        
        if evnt_embedding == "discrete":
            assert dim_E == dim_N, "if event embedding is discrete, then use one dimension for each event type"
            self.evnt_embed = lambda k: (torch.arange(0, dim_E) == k).float()
            
            # output is a dim_N vector, each represent conditional intensity of a type of event
            # L is λ(z(t))
            self.L = nn.Sequential(MLP(dim_c+dim_h, dim_N, dim_hidden, num_hidden, activation), SoftPlus())
            
        else:
            raise Exception('evnt_type must either be discrete or continuous')
         
        # W is jump function: dim_c + dim_E -> dim_h
        self.W = MLP(dim_c + dim_E, dim_h, dim_hidden, num_hidden, activation)
        

        self.backtrace = []

    def forward(self, t, z):
        # f(z(t)) include dc/dt, dh/dt
        c = z[..., :self.dim_c] # 将c和h重新分开，c是前半部分
        h = z[..., self.dim_c:] # h是后半部分
        
        # F: dim_c + dim_h -> dim_c
        dcdt = self.F(z)
        
        # G: dim_c -> dim_h
        dhdt = -self.G(c) * h
        
        return torch.cat((dcdt, dhdt), dim=-1)

    def next_simulated_jump(self, t0, z0, t1):
        # thinalgo
        if not self.evnt_align:
            m = torch.distributions.Exponential(self.L(z0)[..., :self.dim_N].double())
            # next arrival time
            tt = t0 + m.sample()
            tt_min = tt.min()

            if tt_min <= t1:
                dN = (tt == tt_min).float()
            else:
                dN = torch.zeros(tt.shape)

            next_t = min(tt_min, t1)
        else:
            assert t0 < t1

            lmbda_dt = self.L(z0) * (t1 - t0)
            rd = torch.rand(lmbda_dt.shape)
            dN = torch.zeros(lmbda_dt.shape)
            dN[rd < lmbda_dt ** 2 / 2] += 1
            dN[rd < lmbda_dt ** 2 / 2 + lmbda_dt * torch.exp(-lmbda_dt)] += 1

            next_t = t1

        return dN, next_t

    def simulated_jump(self, dN, t, z):
        assert self.jump_type == "simulate", "simulate_jump must be called with jump_type = simulate"
        dz = torch.zeros(z.shape)
        sequence = []

        c = z[..., :self.dim_c]
        for idx in dN.nonzero():
            # find location and type of event
            loc, k = tuple(idx[:-1]), idx[-1]
            ne = int(dN[tuple(idx)])

            for _ in range(ne):
                if self.evnt_embedding == "discrete":
                    # encode of event k
                    kv = self.evnt_embed(k)
                    sequence.extend([(t,) + loc + (k,)])
                elif self.evnt_embedding == "continuous":
                    params = self.L(z[loc])
                    gsmean = params[self.dim_N*(1+self.dim_E*0):self.dim_N*(1+self.dim_E*1)]
                    logvar = params[self.dim_N*(1+self.dim_E*1):self.dim_N*(1+self.dim_E*2)]
                    gsmean_k = gsmean[self.dim_E*k:self.dim_E*(k+1)]
                    logvar_k = logvar[self.dim_E*k:self.dim_E*(k+1)]
                    kv = self.evnt_embed(torch.randn(gsmean_k.shape) * torch.exp(0.5*logvar_k) + gsmean)
                    sequence.extend([(t,) + loc + (kv,)])

                # add to jump
                dz[loc][self.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1))

        self.evnts.extend(sequence)

        return dz

    def next_read_jump(self, t0, t1):
        assert self.jump_type == "read", "next_read_jump must be called with jump_type = read"
        assert t0 != t1, "t0 can not equal t1"

        t = t1
        inf = sys.maxsize
        if t0 < t1:  # forward
            idx = bisect.bisect_right(self.evnts, (t0, inf, inf, inf))
            #print(idx)
            #exit
            if idx != len(self.evnts):
                t = min(t1, torch.tensor(self.evnts[idx][0], dtype=torch.float64))
        else:  # backward
            idx = bisect.bisect_left(self.evnts, (t0, -inf, -inf, -inf))
            if idx > 0:
                t = max(t1, torch.tensor(self.evnts[idx-1][0], dtype=torch.float64))

        assert t != t0, "t can not equal t0"
        return t

    def read_jump(self, t, z):
        assert self.jump_type == "read", "read_jump must be called with jump_type = read"
        dz = torch.zeros(z.shape)

        inf = sys.maxsize
        lid = bisect.bisect_left(self.evnts, (t, -inf, -inf, -inf))
        rid = bisect.bisect_right(self.evnts, (t, inf, inf, inf))

        c = z[..., :self.dim_c]
        for evnt in self.evnts[lid:rid]:
            # find location and type of event
            loc, k = evnt[1:-1], evnt[-1]

            # encode of event k
            kv = self.evnt_embed(k)

            # add to jump
            dz[loc][self.dim_c:] += self.W(torch.cat((c[loc], kv), dim=-1))

        return dz
