from transchannel import *
import unittest

#============================================================================
#============================================================================
## Some tests for global attributes
#============================================================================
#============================================================================
class SimplisticTest(unittest.TestCase):
    def test(self):
        self.assertEqual(5,5)
        self.assertNotEqual(4,5)
unittest.main()



