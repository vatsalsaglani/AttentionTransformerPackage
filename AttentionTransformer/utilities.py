import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import inspect
import tqdm
import contextlib
import os



@contextlib.contextmanager
def redirect_to_tqdm():

    '''
    Keep tqdm.write as the default print 
    '''

    old_print = print
    def new_print(*args, **kwargs):
        
        try:
            tqdm.tqdm.write(*args, **kwargs)

        except:
            old_print(*args, **kwargs)

    
    try:

        inspect.builtins.print = new_print

        yield

    finally:

        inspect.builtins.print = old_print


def tqdm_with_redirect(*args, **kwargs):
    '''
    Single bar for TQDM with print statements
    '''

    with redirect_to_tqdm():

        for x in tqdm.tqdm(*args, **kwargs):

            yield x



def device():
    '''
    keeping code device agnostic 
    '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_model_parameters(model):

    ''' Count of parameters which can be trained in the model defination '''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)






    



