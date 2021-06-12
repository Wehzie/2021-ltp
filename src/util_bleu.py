

"""
Translation metrics.
"""

from nltk.translate import bleu_score


def get_bleu(ht: list, mt, n_gram=4) -> float:
     """
     :param hum: human translation
     :param ml: machine translation
     :param n_gram: number of words considered for BLEU
     :returns: BLEU-N score
     """
     # default is BLEU-4 using 4-grams
     weights = tuple([1 / n_gram for i in range(n_gram)])
     # smoothing function avoids BLEU=0 when no n-grams are found
     # Chen and Cherry developed multiple functions
     # method1 is chosen arbitrarily
     chencherry = bleu_score.SmoothingFunction()

     return bleu_score.sentence_bleu([ht], mt, weights=weights,
          smoothing_function=chencherry.method1)

     
if __name__ == "__main__":
     from pathlib import Path
     import pandas as pd
     train = Path("data/europarl_train.csv")
     df = pd.read_csv(train)

     # bleu score for each human-machine translation pair
     df["bleu"] = df.apply(
          lambda row: get_bleu(
               row["Human"],
               row["Automated"]),
               axis=1)

     # maximum, minimum and mean bleu score in corpus
     print(
     df["bleu"].max(),
     df["bleu"].min(),
     df["bleu"].mean()
     )