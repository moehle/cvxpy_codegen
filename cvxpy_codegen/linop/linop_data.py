from cvxpy_codegen.param.expr_data import ExprData, LINOP_COUNT, CONST_ID
import scipy.sparse as sp
from cvxpy_codegen.linop.linops.linops import GET_LINOPDATA # TODO put this into __init__.py
from cvxpy_codegen.utils.utils import spzeros



class VarData(ExprData):
    def __init__(self, expr):
        self.type = 'var'
        self.id = expr.data
        self.name = 'var%d' % self.id # TODO
        #self.name = expr.name()
        self.arg_data = []
        self.size = expr.size
        self.length = expr.size[0] * expr.size[1]
        self.sparsity = sp.csr_matrix(sp.eye(self.length, dtype=bool))
        self.var_ids = [self.id]
        self.template_vars = dict()
        self.storage = self

    #@property
    #def vars(self):
    #    return [self]

    #@property
    #def var(self):
    #    return self



class LinOpData(ExprData):
    def __init__(self, expr, arg_data):
        super(LinOpData, self).__init__(expr, arg_data)
        self.type = 'linop'
        self.opname = expr.type
        self.name = 'linop%d' % LINOP_COUNT.get_count()
        #self.vars = set().union(*[a.vars for a in arg_data])
        self.data = expr.data
        self.args = arg_data
        self.coeffs = dict()

        # Get leaf vars for this linop.
        #self.vars = []
        #self.has_offset = False
        self.var_ids = []
        for a1 in self.args:
            for vid in a1.var_ids:
                #if vid is CONST_ID:
                #    self.has_offset = True
                #else:
                if vid not in self.var_ids:
                    #self.vars += [var]
                    self.var_ids += [vid]
        #TODO can we not do something like this:
        #self.vars = set().union(*[a.vars for a in arg_data])


        # Get the coefficient for each var
        #print(self.args)
        for vid in self.var_ids:
            coeff_args = []
            for arg in self.args:
                if vid in arg.var_ids:
                    if isinstance(arg, LinOpData):
                        coeff_args += [arg.coeffs[vid]]
                    else:
                        coeff_args += [arg]
            coeff_props = GET_LINOPDATA[self.opname](self, coeff_args, vid)
            coeff = LinOpCoeffData(self, coeff_args, vid, **coeff_props)
            self.coeffs.update({vid : coeff})

        #print("\n")
        #print(self.name)
        #print(self.opname)
        #print(self.args)
        ##c = [c.sparsity for c in self.coeffs.values()]
        #print()



    def get_matrix(self, sym_data):
        coeff_height = self.size[0] * self.size[1]
        mat = spzeros(coeff_height, sym_data.x_length, dtype=bool)
        for vid, coeff in self.coeffs.items():
            if not (vid == CONST_ID): # TODO
                start = sym_data.var_offsets[vid]
                coeff_width = coeff.sparsity.shape[1]
                pad_left = start
                pad_right = sym_data.x_length - coeff_width
                #print('\n')
                #print(spzeros(coeff_height, pad_left, dtype=bool).shape)
                #print(coeff.sparsity.shape)
                #print(coeff.sparsity)
                #print(spzeros(coeff_height, pad_right, dtype=bool).shape)
                #print(coeff.macro_name)
                #print(coeff.name)
                #print(coeff.size)
                mat += sp.hstack([spzeros(coeff_height, pad_left, dtype=bool),
                                  coeff.sparsity,
                                  spzeros(coeff_height, pad_right, dtype=bool)])
                # TODO spzeros should be replaced
        return mat




                



class LinOpCoeffData:
    def __init__(self, linop, args, vid,
                 sparsity=None,
                 inplace=False,
                 macro_name=None,
                 work_int=0,
                 work_float=0,
                 work_coeffs=0,
                 data=None,
                 copy_arg=0):
        self.inplace = inplace
        self.macro_name = macro_name
        self.work_int = work_int
        self.work_float = work_float
        self.work_coeffs = work_coeffs
        self.data = data
        self.args = args
        #print(linop)
        #print(var)

        # TODO combine this (and other things) with ExprData?
        if sparsity == None:
            sparsity = sp.csr_matrix(np.full(expr.size, True, dtype=bool))
        self.sparsity = sparsity

        #if isinstance(vid, VarData):
        #    self.name = linop.name + '_' + var.name
        #else:
        #    self.name = linop.name + '_const'
        self.name = linop.name + '_var' + str(vid) # TODO
        self.type = 'coeff'
        self.vid = vid
        self.size = linop.size
        self.copy_arg = copy_arg

        has_const_or_param = \
                any([a.type =='const' or
                     a.type =='param' or
                     a.type =='var' for a in args])
        if inplace and has_const_or_param:
            self.make_copy = True
        else:
            self.make_copy = False
        self.data = data


    @property
    def storage(self):
        if self.inplace and not self.make_copy:
            return self.args[self.copy_arg].storage
        else:
            return self

    def force_copy(self):
        self.make_copy = True


#        # Constant subsumes Parameter:
#        has_c = any([type(a)=='const' or type(a)=='param' for a in arg_data])
#        if inplace and has_c:
#            self.make_copy = True
#        else:
#            self.make_copy = False
#    def force_copy(self):
#        self.make_copy = True
