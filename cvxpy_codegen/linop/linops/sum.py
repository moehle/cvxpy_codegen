def getdata_sum_lo(linop, args, var):
    #print("\n\nSUM")
    #print(linop.name)
    #print(args[0].name)
    #print(args[0].sparsity.shape)
    #print(args[1].name)
    #print(args[1].sparsity.shape)
    #print("\n")
    #print(linop.name)
    #print(var.name)
    #print("SUM")
    #for a in args:
    #  print("arg:")
    #  print(a.sparsity)
    #  print("result:")
    #print(sparsity)
    #print(sparsity.shape)
    #print(linop.size)

    if len(args) == 1:
        return { "sparsity"    : args[0].sparsity,
                 "work_int"    : 0,
                 "work_float"  : 0,
                 'inplace'     : True,
                 "macro_name"  :'null'}
    else:
        work_coeffs = len(linop.args) # This is a varargs linop.
        #print([a.sparsity for a in args])
        sparsity = sum([a.sparsity for a in args])
        #print(type(args))
        #print(sparsity)
        #raise Exception('this')
        work_int    = sparsity.shape[1]
        work_float  = sparsity.shape[1]
        return { "sparsity"    : sparsity,
                 "work_int"    : work_int,
                 "work_float"  : work_float,
                 "work_coeffs" : work_coeffs,
                 "macro_name"  :'sum'}
