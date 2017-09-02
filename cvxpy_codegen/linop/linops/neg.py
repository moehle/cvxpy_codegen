def getdata_neg_lo(linop, args, var):
    return { 'macro_name'  : "neg",
             'sparsity'    : args[0].sparsity,
             'inplace'     : True }
