# stream.pyx

import cython
#cimport numpy as np


    # hot loop
def merge(x,
          x_inds,
          int x_len,
          int i,
          y,
          y_inds,
          int y_len,
          int j,
          out,
          out_inds,
          int out_len)
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
