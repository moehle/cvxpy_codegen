from cvxpy_codegen.param.expr_data import ExprData, LINOP_COUNT, CONST_ID
#from cvxpy_codegen.linop.linops.linops import GET_LINOPDATA # TODO put this into __init__.py
from cvxpy_codegen.atoms.atoms import get_coeff_data, get_atom_data_from_linop # TODO put this into __init__.py
#from cvxpy_codegen.atoms.atoms import * # TODO put this into __init__.py
from cvxpy_codegen.utils.utils import spzeros # TODO rm
import scipy.sparse as sp
from cvxpy_codegen.linop.linop_coeff_data import LinOpCoeffData



class VarData(ExprData):
    def __init__(self, expr):
        self.type = 'var'
        self.id = expr.data
        self.name = 'var%d' % self.id
        self.arg_data = []
        self.size = expr.size
        self.length = expr.size[0] * expr.size[1]
        self.sparsity = sp.csr_matrix(sp.eye(self.length, dtype=bool))
        self.var_ids = {self.id}
        self.storage = self # Where is the coefficient stored in C?



class LinOpData(ExprData):
    def __init__(self, expr, arg_data):
        super(LinOpData, self).__init__(expr, arg_data)
        self.type = 'linop'
        self.opname = expr.type
        self.name = 'linop%d' % LINOP_COUNT.get_count()
        self.data = expr.data
        self.args = arg_data
        self.coeffs = dict()
        self.var_ids = set().union(*[a.var_ids for a in arg_data])
        self.has_offset = True if CONST_ID in self.var_ids else False
        self.var_ids.discard(CONST_ID)

        # Get the coefficient for each variable.
        for vid in self.var_ids:
            coeff_args = []
            for arg in self.args:
                if vid in arg.var_ids:
                    if isinstance(arg, LinOpData):
                        coeff_args += [arg.coeffs[vid]]
                    else:
                        coeff_args += [arg]
            coeff = get_coeff_data(self, coeff_args, vid) 
            self.coeffs.update({vid : coeff})

        # Get the expression for the offset vector.
        if self.has_offset:
            offset_args = []
            for arg in self.args:
                if CONST_ID in arg.var_ids:
                    if isinstance(arg, LinOpData):
                        offset_args += [arg.offset_expr]
                    else:
                        offset_args += [arg]
            self.offset_expr = get_atom_data_from_linop(self, offset_args)


    def force_copy(self):
        for c in self.coeffs:
            c.force_copy()


    def get_matrix(self, sym_data):
        coeff_height = self.size[0] * self.size[1]
        mat = spzeros(coeff_height, sym_data.x_length, dtype=bool)
        mat = sp.csc_matrix((coeff_height, sym_data.x_length), dtype=bool)
        for vid, coeff in self.coeffs.items():
            if not (vid == CONST_ID):
                start = sym_data.var_offsets[vid]
                coeff_width = coeff.sparsity.shape[1]
                pad_left = start
                pad_right = sym_data.x_length - coeff_width
                mat += sp.hstack([sp.csc_matrix((coeff_height, pad_left), dtype=bool),
                                  coeff.sparsity,
                                  sp.csc_matrix((coeff_height, pad_right), dtype=bool)])
        return mat
