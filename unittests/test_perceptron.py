""" This module implements unittest for the Perceptron class."""
from unittest import TestCase

from perceptron import Perceptron


class TestPerceptron(TestCase):
    """ This module implements unittest for the Perceptron class."""

    def setUp(self):
        """Set up Perceptron model to be used in the unittests."""
        self.model = Perceptron(0.5, 20)

    def test_fit(self):
        """ Test that the perceptron only accepts objects with attribute "len" as
        input."""
        with self.assertRaises(TypeError):
            self.model.fit(0, 0)
