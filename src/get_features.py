import benepar
from numpy import average
import spacy
from nltk import Tree, acyclic_breadth_first
import pandas as pd

#benepar.download('benepar_en3')

def compute_density_metrics(sent):
    densities = {'DET':0, 'AUX':0,'CONJ':0, 'PRON':0, 'ADP':0, 'PUNCT':0, 'FUNC':0}
    
    if len(sent) == 0:
        return densities
    
    for w in sent:
        if w.pos_ in densities.keys():
            densities[w.pos_] += 1
        elif w.pos_ == 'SCONJ':
            densities['CONJ'] += 1
        elif w.pos_ == 'PART':
            densities['FUNC'] += 1
    
    densities['FUNC'] += (densities['DET'] + densities['AUX'] + densities['CONJ'] + densities['PRON'] + densities['ADP'] + densities['PUNCT'])
    cont_words = len(sent) - densities['FUNC'] 
    
    if cont_words > 0 :       
        densities = {k: (v/cont_words/len(sent)) for k,v in densities.items()}
    
    return densities

def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth

def get_const_size(sent, label):
    max_s = 0
    const_num = 0
    average = 0
    for const in sent._.constituents:
        if str(const._.labels) == "(\'" + label + '\',)':
            const_num += 1
            size = len(const)
            average += size
            if max_s < size : max_s = size
    if const_num > 0 : average /= const_num
    return max_s, average



def compute_constituents_metrics(sent):
    constituents = {'S_len': len(sent), 'Depth': walk_tree(sent.root,1)}
    constituents['NP_max'], constituents['NP_avg'] = get_const_size(sent, "NP")
    constituents['PP_max'], constituents['PP_avg'] = get_const_size(sent, "PP")
    constituents['ADVP_max'], constituents['ADVP_avg'] = get_const_size(sent, "ADVP")
    constituents['ADJP_max'], constituents['ADJP_avg'] = get_const_size(sent, "ADJP")
    return constituents


def np_val(x):
    (a) = x._.labels
    try:
        label = a[0]
    except IndexError:
        label = "None"
    if label == "NP":
        return 1
    else:
        return 0


def compute_parse_tree_metrics(sent):
    constituent_iter = iter(sent._.constituents)
    count_left = 0
    count_right = 0
    count_left_NP = 0
    count_right_NP = 0
    for x in constituent_iter:
        if x._.parent != None:
            if list(x)[len(list(x)) - 1] == list(x._.parent)[len(list(x._.parent)) - 1]:
                count_right_NP = count_right_NP + np_val(x)
                count_right = count_right + 1
            if list(x)[0] == list(x._.parent)[0]:
                count_left_NP = count_left_NP + np_val(x)
                count_left = count_left + 1

    parse_tree = {'right_branches_all': count_right/len(sent), 'left_branches_all': count_left/len(sent),
                  'right_branches_NP': count_right_NP / len(sent), 'left_branches_NP': count_left_NP / len(sent),
                  'branching_index_all': count_right - count_left}
    return parse_tree


def get_features(doc, label, original):
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
    