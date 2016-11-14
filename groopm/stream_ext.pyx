# stream.pyx

import cython
#cimport numpy as np


# hot loop
def merge(int x_len,
          x,
          x_inds,
          int y_len,
          y,
          y_inds,
          int out_len,
          out,
          out_inds,
          int i,
          int j):
    for k in range(out_len):
        if j < y_len  and (i==x_len or y[j] < x[i]):
            out[k] = y[j]
            out_inds[k] = y_inds[j]
            j += 1
        else:
            #assert pos_buff < buffl
            out[k] = x[i]
            out_inds[k] = x_inds[i]
            i += 1

        
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
