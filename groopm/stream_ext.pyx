# stream_ext.pyx

import cython
#cimport numpy as np


# hot loop
def merge(x_len,
          x,
          x_inds,
          y_len,
          y,
          y_inds,
          out_len,
          out,
          out_inds):
    i = 0
    j = 0
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
    return (i, j)
        
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
