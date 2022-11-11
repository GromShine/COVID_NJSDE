import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from adjoint import odeint_adjoint as odeint
from numbers import Number
import inspect

# taken fron ricky chen
def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

# create the outdir
def create_outpath(dataset):
    path = os.getcwd()
    pid = os.getpid()

    wsppath = os.path.join(path, 'workspace')
    if not os.path.isdir(wsppath):
        os.mkdir(wsppath)

    outpath = os.path.join(wsppath, 'dataset_'+dataset + '_' + 'pid_'+str(pid))
    assert not os.path.isdir(outpath), 'output directory already exist (process id coincidentally the same), please retry'
    os.mkdir(outpath)

    return outpath


def visualize(outpath, tsave, trace, lmbda, tsave_, trace_, grid, lmbda_real, tse, batch_id, itr, 
              tsave_simu = None,gsmean=None, gsvar=None, scale=1.0, appendix=""):
    # torch.Size([1815, 12, 12])
    for sid in range(lmbda.shape[1]):
        fig = plt.figure(figsize=(20, 10), facecolor='white')
        axe = plt.gca()
        axe.set_title('Point Process Modeling')
        axe.set_xlabel('time')
        axe.set_ylabel('dimension')
        axe.set_ylim(0, 12.0)
        axe2 = axe.twinx()
        axe2.set_ylabel('intensity')
        axe2.set_ylim(-0.5, 2)
        '''
        # plot the state function
        if (tsave is not None) and (trace is not None):
            for dat in list(trace[:, sid, :].detach().numpy().T):
                plt.plot(tsave.numpy(), dat, linewidth=0.3)

        '''
        
        '''
        # plot the state function (backward trace)
        if (tsave_ is not None) and (trace_ is not None):
            for dat in list(trace_[:, sid, :].detach().numpy().T):
                plt.plot(tsave_.numpy(), dat, linewidth=0.2, linestyle="dotted", color="black")
        '''
        
        #'''
        # plot the intensity function
        if (grid is not None) and (lmbda_real is not None):
            axe2.plot(grid.numpy(), lmbda_real[sid], linewidth=1.0, color="gray")
        axe2.plot(tsave.numpy(), lmbda[:, sid, :].detach().numpy(), linewidth=0.7)
        #'''
        
        #'''
        if tse is not None:
            tse_current = [evnt for evnt in tse if evnt[1] == sid]
            # continue...
            if tsave_simu!=None:
                tevnt = np.array([tsave_simu[evnt[0]] for evnt in tse_current])
            else:
                tevnt = np.array([tsave[evnt[0]] for evnt in tse_current])
            kevnt = np.array([evnt[2] if not (type(evnt[2]) == list) else evnt[2][0] for evnt in tse_current])
            axe.scatter(tevnt, kevnt*scale, 1.5)
        #'''
        
        '''
        # plot the gaussian mean
        if gsmean is not None:
            if gsvar is not None:
                for mean, var in zip(list(gsmean[:, sid, :, 0].detach().numpy().T), list(gsvar[:, sid, :, 0].detach().numpy().T)):
                    plt.fill(np.concatenate([tsave.numpy(), tsave.numpy()[::-1]]),
                             np.concatenate([scale * (mean - 1.9600 * np.sqrt(var)),
                                             scale * (mean + 1.9600 * np.sqrt(var))[::-1]]),
                             alpha=0.1, fc='b', ec='None')

            for mean in list(gsmean[:, sid, :, 0].detach().numpy().T):
                plt.plot(tsave.numpy(), scale * mean, linewidth=1.0, linestyle="dotted", color="black")
        '''

        plt.savefig(outpath + '/{}_{:03d}_{:04d}.png'.format(appendix,batch_id[sid], itr), dpi=360)
        fig.clf()
        plt.close(fig)


# this function takes in a time series and create a grid for modeling it
# it takes an array of sequences of three tuples, and extend it to four tuple
def create_tsave(tmin, tmax, dt, evnts_raw, evnt_align=False):
    # param tmin: min time of sequence
    # param tmax: max time of the sequence
    # param dt: step size
    # param evnts_raw: (事件时间,事件发生在哪一维度/地区,事件类型)
    # param evnt_align: whether to round the event time up to the next grid point
    # return tsave: the time to save state in ODE simulation
    # return gtid: grid time id
    # return evnts: tuple (rounded_time, ...)
    # return tse: tuple (event_time_id, ...)

    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    # 如果align是true,对时间点进行取整处理,否则不变,原来的时间点序列
    # evnts: (事件时间,事件发生在哪一维度/地区,事件类型)
    evnts = [(tc(evnt[0]),) + evnt[1:] for evnt in evnts_raw if tmin < tc(evnt[0]) < tmax]
    
    # 时间网格,步长0.05,from tmin to tmax
    tgrid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    
    # 将全部事件的时间取出来单独存放
    tevnt = np.array([evnt[0] for evnt in evnts])
    
    # 将全部事件时间和时间网格合并后去重排序
    tsave = np.sort(np.unique(np.concatenate((tgrid, tevnt))))
    
    # time_to_timeid, 时间去重后的所有时间里的排序位置
    t2tid = {t: tid for tid, t in enumerate(tsave)}

    # gridtime_id
    # 时间网格在去重后所有时间里的排序位置
    gtid = [t2tid[t] for t in tgrid]
    
    # t(ime)s(equence)n(ode)e(vent)
    # 发生事件对应的（时间排序位次,维度,事件类型）tuple
    tse = [(t2tid[evnt[0]],) + evnt[1:] for evnt in evnts]
    

    # 返回全部时间+时间网格去重后的时间, 时间网格排序位置, 原输入时间维度类型元组, 发生事件tuple(时间位置,维度,事件类型元组)
    return torch.tensor(tsave), gtid, evnts, tse

def forward_pass(func, z0, tspan, dt, batch, evnt_align, gs_info=None, type_forecast=[0.0], 
                 predict_first=True, rtol=1.0e-5, atol=1.0e-7, scale=1.0):
#def forward_pass(func, z0, tspan, dt, batch, evnt_align, gs_info=None, type_forecast=[0.0], predict_first=True, rtol=1.0e-5, atol=1.0e-7, scale=1.0):
    
    #print(inspect.getfile(odeint))
    
    # merge the sequences to create a sequence
    # batch里的(事件时间,事件发生在哪一维度/地区,事件类型)
    evnts_raw = sorted([(evnt[0],) + (sid,) + evnt[1:] for sid in range(len(batch)) for evnt in batch[sid]])
    
    # set up grid
    tsave, gtid, evnts, tse = create_tsave(tspan[0], tspan[1], dt, evnts_raw, evnt_align)
    
    
    # same evnts_raw
    func.evnts = evnts

    # convert to numpy array
    # 所有事件时间+网格时间, the time to save state in ODE simulation
    tsavenp = tsave.numpy()

    # func: ODEJumpFunc
    # z0: dim_N*(c0+h0) -> county_num * (n1+n2), 初始化所有维度的z0
    # output trace: t_total * county_num * z_i = t_total * county_num * (c_i+h_i)
    
    trace = odeint(func, z0.repeat(len(batch), 1), tsave, method='jump_adams', rtol=rtol, atol=atol)
    # input: t_total*county_num*(n1+n2)
    # output: t_total*couty_num*dim_N,每个时间节点,每个维度,每个事件类型的lambda
    # torch.Size([1815, 12, 20])
    
    
    #print("before_L")
    #print(trace.size())
    # params Size([1815, 12, 12])
    # torch.Size([2242, 1, 22])
    params = func.L(trace)
    
    # print(lmbda.size())
    # torch.Size([1815, 12, 12])
    # 取出了事件类型总数dim_N的lambda
    lmbda = params[..., :func.dim_N]
    
    # print(tsave.size())
    # torch.Size([1815])
    
    if gs_info is not None:
        lmbda[:, :, :] = torch.tensor(gs_info[0])

    # 通过学到的lambda函数积分求出loglikehood后半部分
    def integrate(tt, ll):
        # 上底加下底乘以高除以2，去掉最后一个的序列，去掉第一个的序列，组成λ的上底和下底
        # torch.Size([1814, 12, 12])
        lm = (ll[:-1, ...] + ll[1:, ...]) / 2.0
        
        # 求出每段的高(时间间隔)
        dts = (tt[1:] - tt[:-1]).reshape((-1,) + (1,)*(len(lm.shape)-1)).float()
        return (lm * dts).sum()

    log_likelihood = -integrate(tsave, lmbda)

    # set of sequences where at least one event has happened, 一个batch大小的set
    seqs_happened = set(sid for sid in range(len(batch))) if predict_first else set()

    if func.evnt_embedding == "discrete":
        et_error = []
        # 对每一个事件tuple(时间排序位次,维度,事件类型)
        for evnt in tse:
            log_likelihood += torch.log(lmbda[evnt])
            
            # 如果当前取出来的事件在我们batch的维度里
            if evnt[1] in seqs_happened:
                
                # 置0
                type_preds = torch.zeros(len(type_forecast))
                
                # 默认的tid和t全是0
                for tid, t in enumerate(type_forecast):
                    
                    # 在所有的事件时间和整点时间里,找出evnt[0]的时间-t放在哪里可以保持原序列不变
                    loc = (np.searchsorted(tsavenp, tsave[evnt[0]].item()-t),) + evnt[1:-1]
                    
                    # 找出lambda强度最大的一项作为预测事件种类
                    type_preds[tid] = lmbda[loc].argmax().item()
                
                # 预测错了就加1
                et_error.append((type_preds != evnt[-1]).float())
            
            # 将当前事件加入到batch维度的set中
            seqs_happened.add(evnt[1])
        
        # 平均每步预测错多少事件类型
        METE = sum(et_error)/len(et_error) if len(et_error) > 0 else -torch.ones(len(type_forecast))

    elif func.evnt_embedding == "continuous":
        gsmean = params[..., func.dim_N*(1+func.dim_E*0):func.dim_N*(1+func.dim_E*1)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        logvar = params[..., func.dim_N*(1+func.dim_E*1):func.dim_N*(1+func.dim_E*2)].view(params.shape[:-1]+(func.dim_N, func.dim_E))
        var = torch.exp(logvar)

        if gs_info is not None:
            gsmean[:, :, :] = torch.tensor(gs_info[1])
            var[:, :, :] = torch.tensor(gs_info[2])

        def log_normal_pdf(loc, k):
            const = torch.log(torch.tensor(2.0*np.pi))
            return -0.5*(const + logvar[loc] + (gsmean[loc] - func.evnt_embed(k))**2.0 / var[loc])

        et_error = []
        for evnt in tse:
            log_gs = log_normal_pdf(evnt[:-func.dim_N], evnt[-func.dim_N:]).sum(dim=-1)
            log_likelihood += logsumexp(lmbda[evnt[:-func.dim_N]].log() + log_gs, dim=-1)
            if evnt[1] in seqs_happened:
                # mean_pred embedding
                mean_preds = torch.zeros(len(type_forecast), func.dim_E)
                for tid, t in enumerate(type_forecast):
                    loc = (np.searchsorted(tsavenp, tsave[evnt[0]].item()-t),) + evnt[1:-func.dim_N]
                    mean_preds[tid] = ((lmbda[loc].view(func.dim_N, 1) * gsmean[loc]).sum(dim=0) / lmbda[loc].sum()).detach()
                et_error.append((mean_preds - func.evnt_embed(evnt[-func.dim_N:])).norm(dim=-1))
            seqs_happened.add(evnt[1])

        METE = sum(et_error)*scale/len(et_error) if len(et_error) > 0 else -torch.ones(len(type_forecast))

    if func.evnt_embedding == "discrete":
        return tsave, trace, lmbda, gtid, tse, -log_likelihood, METE
    elif func.evnt_embedding == "continuous":
        return tsave, trace, lmbda, gtid, tse, -log_likelihood, METE, gsmean, var


def poisson_lmbda(tmin, tmax, dt, lmbda0, TS):
    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)

    lmbda = []
    for _ in TS:
        lmbda.append(lmbda0 * np.ones(grid.shape))

    return lmbda


def exponential_hawkes_lmbda(tmin, tmax, dt, lmbda0, alpha, beta, TS, evnt_align=False):
    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    cl = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    t2tid = {t: tid for tid, t in enumerate(grid)}

    lmbda = []
    kernel = alpha * np.exp(-beta * np.arange(0.0, 10.0/beta, dt))

    for ts in TS:
        vv = np.zeros(grid.shape)
        for record in ts:
            vv[t2tid[cl(record[0])]] = np.exp(-beta * (cl(record[0]) - tc(record[0])))
        lmbda.append(lmbda0 + np.convolve(kernel, vv)[:grid.shape[0]])

    return lmbda


def powerlaw_hawkes_lmbda(tmin, tmax, dt, lmbda0, alpha, beta, sigma, TS, evnt_align=False):
    cl = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)
    t2tid = {t: tid for tid, t in enumerate(grid)}

    lmbda = []
    if evnt_align:
        kernel_grid = np.arange(dt, 10.0 + 10.0*sigma, dt)
        kernel = np.concatenate(([0], alpha * (beta/sigma) * (kernel_grid / sigma)**(-beta-1) * (kernel_grid > sigma)))
        for ts in TS:
            vv = np.zeros(grid.shape)
            for record in ts:
                vv[t2tid[cl(record[0])]] = 1.0
            lmbda.append(lmbda0 + np.convolve(kernel, vv)[:grid.shape[0]])
    else:
        for ts in TS:
            vv = np.zeros(grid.shape)
            for record in ts:
                lo = t2tid[cl(min(record[0]+sigma, 100.0))]
                hi = t2tid[cl(min(record[0]+10.0+10.0*sigma, 100.0))]
                vv[lo:hi] += alpha * (beta/sigma) * ((grid[lo:hi]-record[0]) / sigma)**(-beta-1)
            lmbda.append(lmbda0 + vv)

    return lmbda


def self_inhibiting_lmbda(tmin, tmax, dt, mu, beta, TS, evnt_align=False):
    if evnt_align:
        tc = lambda t: np.round(np.ceil((t-tmin) / dt) * dt + tmin, decimals=8)
    else:
        tc = lambda t: t

    grid = np.round(np.arange(tmin, tmax+dt, dt), decimals=8)

    lmbda = []
    for ts in TS:
        log_lmbda = mu * (grid-tmin)
        for record in ts:
            log_lmbda[grid > tc(record[0])] -= beta
        lmbda.append(np.exp(log_lmbda))

    return lmbda


def read_timeseries(filename, num_seqs=sys.maxsize):
    with open(filename) as f:
        seqs = f.readlines()[:num_seqs]
    return [[(float(t), 0) for t in seq.split(';')[0].split()] for seq in seqs]
