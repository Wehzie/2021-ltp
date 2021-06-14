import benepar
import spacy
import pandas as pd
import os
import sys
from pathlib import Path

#benepar.download('benepar_en3')

sys.path.append(os.getcwd())
from src.utils import *

def get_features(doc, label, original) -> dict:
    # get the first sentence only
    sent = list(doc.sents)[0]

    # make dict of all features 
    example = {'label': label, 'original': original, 'text': sent.text}
    example.update(compute_density_metrics(sent))
    example.update(compute_constituents_metrics(sent))
    example.update(compute_parse_tree_metrics(sent))
    return example

def get_corpus_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Speed: 51 seconds for 100 rows.
    """
    list_of_examples = []
    size = len(df.index)
    
    count=0
    for _, line in df.iterrows():
        if count % 10000 == 0:
            print('Computed features for {:.4f}% of lines'.format(count/size*100))
            
        # get features of the human-translated example 
        example_human = get_features(nlp(line['Human']), 'human', line['Original'])
        list_of_examples.append(example_human)
        
        # get features of the automated translation example
        example_auto = get_features(nlp(str(line['Automated']).replace("&#39;", "'")), 'automated', line['Original'])
        list_of_examples.append(example_auto)
        count += 1

    return pd.DataFrame(list_of_examples)

def format(df: pd.DataFrame) -> pd.DataFrame:
    """
    df.head():
    Index   Original    Translator  Translation

                        Human
                        Automated
                        Human
                        Automated
                        ...
                        
    """

    # separate rows for human and machine translations
    df = df.melt(id_vars=["Original"], 
        value_vars=["Human", "Automated"],
        var_name="Translator", 
        value_name="Translation")
    # manually name index to "Index"
    df.index.rename("Index", inplace=True)
    # sort data frame
    df = df.sort_values(by=["Original", "Index"])
    return df

def get_corpus_features_list_comp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Speed: 50 seconds for 100 rows.
    """
    df = format(df)
    out = [ get_features(nlp(str(translation).replace("&#39;", "'")), label, original) 
            for label, translation, original in 
            zip(df["Translator"], df["Translation"], df["Original"]) ]
    return pd.DataFrame.from_records(out)

if __name__ == "__main__":
    nlp = spacy.load('en_core_web_md')

    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    # dev data
    path = Path('data/dev/europarl_dev.csv')
    df = pd.read_csv(path, index_col = 0, header = 0, nrows=100)
    get_corpus_features(df).to_csv(Path('data/dev/features_dev_t.csv'))
        
    # test data
    path = Path('data/test/europarl_test.csv')
    df = pd.read_csv(path, index_col = 0, header = 0)
    get_corpus_features(df).to_csv(Path('data/test/features_test.csv'))