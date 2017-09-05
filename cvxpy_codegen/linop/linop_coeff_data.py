from cvxpy_codegen.param.expr_data import ExprData


class LinOpCoeffData(ExprData):
    def __init__(self, linop_data, arg_data, vid,
                 sparsity=None,
                 inplace=False,
                 macro_name=None,
                 work_int=0,
                 work_float=0,
                 work_coeffs=0,
                 data=None,
                 copy_arg=0):
        super(LinOpCoeffData, self).__init__(linop_data, arg_data,
                                             sparsity=sparsity)
        self.inplace = inplace
        self.macro_name = macro_name
        self.work_int = work_int
        self.work_float = work_float
        self.work_coeffs = work_coeffs
        self.data = data
        self.name = linop_data.name + '_var' + str(vid)
        self.type = 'coeff'
        self.vid = vid
        self.size = linop_data.size
        self.copy_arg = copy_arg
        has_const_or_param = any([a.type =='const' or
                                  a.type =='param' or
                                  a.type =='var' for a in arg_data])
        if inplace and has_const_or_param:
            self.make_copy = True
        else:
            self.make_copy = False
        self.data = data
        self.name = self.storage.name


    @property
    def storage(self):
        if self.inplace and not self.make_copy:
            return self.args[self.copy_arg].storage
        else:
            return self

    def force_copy(self):
        self.make_copy = True

