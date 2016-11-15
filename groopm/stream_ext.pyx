# stream_ext.pyx

import cython
#cimport numpy as np


# hot loop
def merge(x,
          x_inds,
          y,
          y_inds,
          out,
          out_inds):
    i = 0
    j = 0
    x_len = len(x)
    y_len = len(y)
    out_len = len(out)
    #assert y_len + x_len >= out_len
    for k in range(out_len):
        if j < y_len  and (i==x_len or y[j] < x[i]):
            out[k] = y[j]
            out_inds[k] = y_inds[j]
            j += 1
        else:
            #assert i < x_len
            out[k] = x[i]
            out_inds[k] = x_inds[i]
            i += 1
    #assert i + j == out_len
    return (i, j)
        
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
