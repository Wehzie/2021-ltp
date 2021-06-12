import benepar
import spacy
from numpy import average
from nltk import Tree, acyclic_breadth_first
import pandas as pd
import os
import sys

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

    

if __name__ == "__main__":
    
    nlp = spacy.load('en_core_web_md')
    
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    path = 'data/dev/europarl_dev.csv'
    data = pd.read_csv(path, index_col= 0, header = 0)
    
    list_of_examples = []
    count = 0
    size = len(data.index)
    
    for _, line in data.iterrows():
        
        # if count == 100:
        #     break
        
        if count % 10000 == 0:
            print('Computed features for {:.4f}% of lines'.format(count/size*100))
            
        # get features of the human-translated example 
        example_human = get_features(nlp(line['Human']), 'human', line['Original'])
        list_of_examples.append(example_human)
        
        # get features of the automated translation example
        example_auto = get_features(nlp(str(line['Automated']).replace("&#39;", "'")), 'automated', line['Original'])
        list_of_examples.append(example_auto)
        count += 1
        
    features_df = pd.DataFrame(list_of_examples)
    features_df.to_csv('data/dev/features_dev.csv')
    