import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint_adjoint
from numbers import Number
import torch.nn as nn


def visualize(outpath, tsave, trace, lmbda, tsave_, trace_, grid, lmbda_real, tse, 
              batch_id, itr, gsmean=None, gsvar=None, scale=1.0, appendix=""):
    for sid in range(lmbda.shape[1]):
        fig = plt.figure(figsize=(20, 20), facecolor='white')
        axe = plt.gca()
        axe.set_title('Point Process Modeling')
        axe.set_xlabel('time')
        axe.set_ylabel('intensity')
        axe.set_ylim(-3.5, 3.5)
        axe2 = axe.twinx()
        axe2.set_ylabel('intensity2')
        axe2.set_ylim(-1.5, 1.5)
        # plot the state function
        
        if (tsave is not None) and (trace is not None):
            for dat in list(trace[:, sid, :].detach().numpy().T):
                plt.plot(tsave.numpy(), dat, linewidth=0.3)

        
        # plot the state function (backward trace)
        if (tsave_ is not None) and (trace_ is not None):
            for dat in list(trace_[:, sid, :].detach().numpy().T):
                plt.plot(tsave_.numpy(), dat, linewidth=0.2, linestyle="dotted", color="black")
        
        
        '''
        # plot the intensity function
        if (grid is not None) and (lmbda_real is not None):
            plt.plot(grid.numpy(), lmbda_real[sid], linewidth=1.0, color="gray")
        plt.plot(tsave.numpy(), lmbda[:, sid, :].detach().numpy(), linewidth=0.7)
        '''
        
        # plot the intensity function
        if (grid is not None) and (lmbda_real is not None):
            axe2.plot(grid.numpy(), lmbda_real[sid], linewidth=1.0, color="gray")
        axe2.plot(tsave.numpy(), lmbda[:, sid, :].detach().numpy(), linewidth=0.7)
        
        
        '''
        if tse is not None:
            tse_current = [evnt for evnt in tse if evnt[1] == sid]
            # continue...
            tevnt = np.array([tsave[evnt[0]] for evnt in tse_current])
            kevnt = np.array([evnt[2] if not (type(evnt[2]) == list) else evnt[2][0] for evnt in tse_current])
            plt.scatter(tevnt, kevnt*scale, 1.5)
        '''
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
        
        plt.savefig(outpath + '/{:04d}_{:03d}_{}.png'.format(itr,batch_id[sid],appendix), dpi=480)
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
    # 发生事件对应的（时间位置,维度,事件类型）tuple
    tse = [(t2tid[evnt[0]],) + evnt[1:] for evnt in evnts]

    #返回全部时间+时间网格去重后的时间, 时间网格排序位置, 原输入时间维度类型元组, 发生事件的对应的（时间位置,维度,事件类型）元组
    return torch.tensor(tsave), gtid, evnts, tse

def forward_pass(func, z0, tspan, dt, batch, evnt_align, A_matrix, gs_info=None, type_forecast=[0.0], 
                 predict_first=True, rtol=1.0e-5, atol=1.0e-7, scale=1.0):
    # merge the sequences to create a sequence
    
    # 取出随机batch里的(事件时间,事件发生在哪一维度/地区,事件类型)
    evnts_raw = sorted([(evnt[0],) + (sid,) + evnt[1:] for sid in range(len(batch)) for evnt in batch[sid]])
    
    # set up grid
    tsave, gtid, evnts, tse = create_tsave(tspan[0], tspan[1], dt, evnts_raw, evnt_align)
    
    #tse,发生事件对应的（时间排序位次,维度,事件类型）tuple
    func.evnts = evnts
    #func里的evnts就是外面的evnts_raw
    
    # convert to numpy array
    tsavenp = tsave.numpy()
    #所有的事件时间和整点时间, the time to save state in ODE simulation
    
    # forward pass
    trace = odeint_adjoint(func, z0.repeat(len(batch), 1), tsave, method='jump_adams', rtol=rtol, atol=atol)
    # input: county_num * (n1+n2)
    # output: t_total*county_num*(n1+n2)
    
    params = func.L(trace)
    
    # input: t_total*county_num*(n1+n2)
    # output: t_total*couty_num*dim_N,每个维度,每个时间节点上、每个事件类型的lambda
    lmbda = params[..., :func.dim_N]
    # print(lmbda.size())
    # torch.Size([1815, 12, 12])
    # 取出了事件类型总数的lambda,我们这里只有一种事件类型
    
    if gs_info is not None:
        lmbda[:, :, :] = torch.tensor(gs_info[0])

    
    # 通过学到的lambda函数积分求出loglikehood后半部分
    def integrate(tt, ll):
        lm = (ll[:-1, ...] + ll[1:, ...]) / 2.0
        dts = (tt[1:] - tt[:-1]).reshape((-1,)+(1,)*(len(lm.shape)-1)).float()
        return (lm * dts).sum()
    
    log_likelihood = -integrate(tsave, lmbda)
        
    
    # set of sequences where at least one event has happened,一个batch大小的set
    seqs_happened = set(sid for sid in range(len(batch))) if predict_first else set()
    
    if func.evnt_embedding == "discrete":
        et_error = []
        for evnt in tse:
            log_likelihood += torch.log(lmbda[evnt])
            if evnt[1] in seqs_happened:
                # 如果当前取出来的事件在我们batch的维度里
                type_preds = torch.zeros(len(type_forecast))
                # 置0
                for tid, t in enumerate(type_forecast):
                    # 默认的tid和t全是0
                    loc = (np.searchsorted(tsavenp, tsave[evnt[0]].item()-t),) + evnt[1:-1]
                    # 在所有的事件时间和整点时间里,找出evnt[0]的时间-t放在哪里可以保持原序列不变
                    type_preds[tid] = lmbda[loc].argmax().item()
                    # 找出lambda强度最大的一项作为预测事件种类
                et_error.append((type_preds != evnt[-1]).float())
                # 预测错了就加1
            seqs_happened.add(evnt[1])
            # 将当前事件加入到batch维度的set中

        METE = sum(et_error)/len(et_error) if len(et_error) > 0 else -torch.ones(len(type_forecast))
        # 平均每步预测错多少事件类型

    #print(log_likelihood,"like")
    #exit
    if func.evnt_embedding == "discrete":
        return tsave, trace, lmbda, gtid, tse, -log_likelihood, METE


def read_timeseries(filename, num_seqs=sys.maxsize):
    with open(filename) as f:
        seqs = f.readlines()[:num_seqs]
    return [[(float(t), 0) for t in seq.split(';')[0].split()] for seq in seqs]
