from cvxpy_codegen.utils.utils import Counter, spzeros
from cvxpy_codegen.param.expr_data import CONST_ID
import scipy.sparse as sp
import numpy as np


CONSTR_COUNT = Counter()


class ConstrData():
    
    def __init__(self, constr, linop, vert_offset):
        self.name = 'constr%d' % CONSTR_COUNT.get_count()
        self.linop = linop
        self.size = linop.size[0] * linop.size[1]
        self.vert_offset = vert_offset

    def get_matrix(self, sym_data):
        return self.linop.get_matrix(sym_data)
