import unittest

from src import util_bleu


class TestMain(unittest.TestCase):
    def test_get_bleu(self):
        """
        BLEU scores should be between 0 and 1.
        """
        ht = "hello today is a very beautiful day".split(" ")
        mt = "hello today is a very pretty day".split(" ")
        n_gram = [2, 4]
        for n in n_gram:
            bleu = util_bleu.get_bleu(ht, mt, n)
            self.assertTrue(0 <= bleu <= 1)


# make tests runnable from the command line
if __name__ == "__main__":
    unittest.main()
