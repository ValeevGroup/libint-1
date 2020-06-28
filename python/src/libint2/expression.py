from . import BasisSet, Engine, BraKet, Operator, _libint2

import re, numpy as np
from collections import namedtuple

class Expression:

  operator = {
    'T' : Operator.kinetic,
    'V' : Operator.nuclear,
    'S' : Operator.overlap,
    '' : Operator.coulomb
  }

  Index = namedtuple("Index", ["basis", "transform"])

  def __init__(self, *, engine=None, charges=None, **kwargs):
    self.index = dict()
    for (idx,basis) in kwargs.items():
      for i in idx:
        self.index[i] = Expression.make_index(i,basis)
    self._engine = engine or Engine(Operator.overlap)
    self._charges = charges

  @staticmethod
  def make_basis(basis):
    if not isinstance(basis,BasisSet):
      basis = BasisSet(basis)
    return basis

  @staticmethod
  def make_index(idx,basis):
    try:
      basis = Expression.make_basis(basis)
      return Expression.Index( basis, None )
    except: pass
    try:
      basis,transform = basis
      basis = Expression.make_basis(basis)
      transform = np.array(transform)
      return Expression.Index(basis,transform)
    except Exception:
      error = "Index %r: invalid basis tuple (%s)" % (idx, basis)
      raise Exception(error) from None
    return basis

  @staticmethod
  def parse(formula):
    bra = r"(?P<bra>[^|]+)"
    ket = r"(?P<ket>[^|]+)"
    op = r"\|((?P<op>.*)\|)?" # | or |OP|
    grammar = bra + op + ket
    #print ("\nformula =",formula)
    f = re.sub(r"\((.*)\)", r"\1", formula.strip()) # remove ()
    match = (re.fullmatch(grammar, f))
    if not match:
      raise Exception("Invalid integral formula %r" % formula)
    #print (formula, match, 'op=%r' % match.group('op'))
    op = (match.group("op") or "").strip()
    op = Expression.operator[op];
    return (
      match.group("bra").replace(" ", ""),
      op,
      match.group("ket").replace(" ", "")
    )

  @staticmethod
  def braket(bra, op, ket):
    braket1 = {
      (1,1) : BraKet.XX,
    }
    braket2 = {
      (1,1) : BraKet.XSXS,
      (2,1) : BraKet.XXXS,
      (1,2) : BraKet.XSXX,
      (2,2) : BraKet.XXXX,
    }
    if Operator.rank(op) == 1:
      return braket1[(len(bra),len(ket))]
    if Operator.rank(op) == 2:
      return braket2[(len(bra),len(ket))]
    assert False

  def compute(self, formula):
    bra,op,ket = self.parse(formula)
    #print (bra,op,ket)
    engine = self._engine
    engine.oper = op
    engine.braket = Expression.braket(bra, op, ket)
    if op == Operator.nuclear:
      engine.set_params(self._charges)
    indices = [ self.index[i] for i in bra+ket ]
    basis = [ index.basis for index in indices ]
    #print (engine.oper, engine.braket, basis)
    V = engine.compute(*basis)
    transforms = []
    axis = []
    for i,idx in enumerate(indices):
      axis.append(i)
      if idx.transform is None: continue
      inner = len(bra+ket)+i
      axis[-1] = inner
      t = ( idx.transform, [inner,i] )
      transforms += t
    if not transforms: return V
    return np.einsum(V, axis, *transforms)
