import scipy.sparse as sp

def getdata_index_lo(linop, args, var):
    sz0, sz1 = args[0].size
    slice0 = linop.data[0]
    slice1 = linop.data[1]
    
    # Get data.
    step0 = 1 if slice0.step==None else slice0.step
    step1 = 1 if slice1.step==None else slice1.step
    data = {'start0' : slice0.start,
            'stop0'  : slice0.stop,
            'step0'  : step0,
            'start1' : slice1.start,
            'stop1'  : slice1.stop,
            'step1'  : step1}

    idxs0 = range(*slice0.indices(sz0))
    idxs1 = range(*slice1.indices(sz1))
    indices = []
    #print("\n\nINDEX")
    #print(sz0)
    #print(sz1)
    #print(slice0)
    #print(slice1)
    #print(idxs0)
    #print(idxs1)
    for idx0 in idxs0:
      for idx1 in idxs1:
        indices += [idx0 + sz0 * idx1]
    #print(indices)
    sp.csr_matrix(args[0].sparsity)[indices,:]
    sparsity = args[0].sparsity[indices, :]


    return { 'sparsity'     : sparsity,
             'work_int'     : 0,
             'work_float'   : 0,
             'macro_name'   : 'index',
             'data'         : data}
