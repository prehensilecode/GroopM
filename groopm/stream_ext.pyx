# stream_ext.pyx

#import cython
cimport numpy as np

# hot loop
def merge(np.ndarray[np.double_t] x,
          np.ndarray[np.int_t] x_inds,
          np.ndarray[np.double_t] y,
          np.ndarray[np.int_t] y_inds,
          np.ndarray[np.double_t] out,
          np.ndarray[np.int_t] out_inds):
    cdef int i = 0
    cdef int j = 0
    cdef int x_len = x.size
    cdef int y_len = y.size
    cdef int out_len = out.size
    cdef int k
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
