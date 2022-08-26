import unittest
from perceptron import perceptron

class Testperceptron(unittest.TestCase):
    def setUp(self):
        self.model = perceptron(0.5, 20)

class TestFit(Testperceptron):
    def test_reject_numbers(self):
        # Make sure that the perceptron only accepts objects with attribute "len" as input.
        with self.assertRaises(TypeError):
            self.model.fit(0, 0)

if __name__ == "__main__":
    unittest.main()