import unittest
import libint2 as libint
import numpy as np
from numpy.linalg import norm

from libint2 import Shell, BasisSet

libint.Engine.num_threads = 1

s = Shell(0, [(1,10)])
p = Shell(1, [(1,10)])

h2o = [
  (8, [  0.00000, -0.07579, 0.00000 ]),
  (1, [  0.86681,  0.60144, 0.00000 ]),
  (1, [ -0.86681,  0.60144, 0.00000 ]),
]

class TestLibint(unittest.TestCase):

  def test_integrals(self):
    self.assertAlmostEqual(norm(libint.kinetic().compute(s,s)), 1.5)
    self.assertAlmostEqual(norm(libint.overlap().compute(s,s)), 1.0)
    self.assertAlmostEqual(norm(libint.nuclear(h2o).compute(s,s)), 14.54704336519)

    self.assertAlmostEqual(norm(libint.coulomb().compute(p,p,s,s)), 1.62867503968)

    self.assertAlmostEqual(
      norm(libint.Engine(libint.Operator.coulomb, braket=libint.BraKet.XXXS).compute(s,s,s)),
      3.6563211198
    )

    basis = [ s, p, s, p ]
    self.assertAlmostEqual(libint.overlap().compute(basis, basis).sum(), 16.0)
    self.assertAlmostEqual(
      norm(libint.coulomb().compute(basis, basis, basis, basis)),
      14.7036075402
    )

  def test_basis(self):
    basis = BasisSet('6-31g', h2o)
    self.assertEqual(len(basis), 9)
    basis.pure = False
    pure = [False]*len(basis)
    self.assertEqual([ s.pure for s in basis], pure)
    basis[0].pure = True
    pure[0] = True
    self.assertEqual([ s.pure for s in basis], pure)

  def test_hf(self):
    from libint2.hf import RHF
    basis = '6-31g'
    rhf = RHF(basis, h2o)
    self.assertAlmostEqual(rhf.energy(), -75.1903033978)

  def test_basis_from_bse(self):
    from libint2.hf import RHF
    bse = libint.basis.load_from_bse('6-31g')
    bse = BasisSet(bse, h2o)
    ref = BasisSet('6-31g', h2o)
    self.assertEqual(len(bse), len(ref))
    self.assertAlmostEqual(
      RHF(bse,h2o).energy(),
      RHF(ref,h2o).energy()
    )

if __name__ == '__main__':
  unittest.main()
