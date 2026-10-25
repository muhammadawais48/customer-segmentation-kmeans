import unittest,tempfile
from pathlib import Path
from src.analysis import analyze
class SegmentationTests(unittest.TestCase):
 def test_end_to_end(self):
  with tempfile.TemporaryDirectory() as d:
   result=analyze(Path('data/customers.csv'),Path(d))
   self.assertEqual(result['customers'],240)
   self.assertEqual(sum(result['cluster_sizes'].values()),240)
   self.assertTrue((Path(d)/'customer_segments.png').exists())
 def test_bad_k(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):analyze(Path('data/customers.csv'),Path(d),1)
if __name__=='__main__':unittest.main()
