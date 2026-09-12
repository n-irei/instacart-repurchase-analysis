import unittest
import numpy as np
from metrics import order_f1, grouped_f1

class MetricTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(order_f1([], []), 1)
        self.assertEqual(order_f1([], [1]), 0)
        self.assertEqual(order_f1([1], []), 0)
        self.assertAlmostEqual(order_f1([], ['None', 1]), 2/3)
        self.assertAlmostEqual(order_f1([1], ['None', 1]), 2/3)
        self.assertEqual(order_f1([], ['none']), 0)
        self.assertEqual(order_f1([1,1], [1]), 1)

    def test_vector_against_independent_set_reference(self):
        rng=np.random.default_rng(773)
        for _ in range(100):
            sizes=rng.integers(0,30,25)
            group=np.repeat(np.arange(25),sizes)
            y=rng.integers(0,2,len(group)); p=rng.random(len(group))
            pn=rng.random(25)
            got=grouped_f1(group,y,p,25,.35,pn,.5)
            expected=[]
            for i in range(25):
                ix=np.flatnonzero(group==i)
                a=ix[y[ix]==1].tolist()
                b=ix[p[ix]>=.35].tolist()
                if not b or pn[i]>=.5: b.append('None')
                expected.append(order_f1(a,b))
            np.testing.assert_allclose(got,expected)

if __name__=='__main__': unittest.main()
