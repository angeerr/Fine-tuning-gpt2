import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import psutil, os

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def draw_loss(lossList,mode,path):
    if mode == 'train':
        loss = [i['train loss'] for i in lossList if 'train loss' in i]
    elif mode == 'test':
        loss = [i['test loss'] for i in lossList if 'test loss' in i]
    else:
        raise NotImplementedError

    plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.xlabel("iterations")
    plt.ylabel(mode+' loss')
    plt.grid(True) 
    if path.endswith('/'):
        plt.savefig(path+mode+'.jpg')
    else:
        plt.savefig(path+'/'+mode+'.jpg')

def get_gpu_memory():
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved()
    return allocated, cached