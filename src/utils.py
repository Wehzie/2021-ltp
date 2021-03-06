import benepar
import pandas as pd
import spacy

# Before usage run the following in your command line:
#       python3 -m spacy download en_core_web_md

# And on the first run of the script uncomment the following line:
# benepar.download('benepar_en3')

def compute_density_metrics(sent) -> dict:
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



def walk_tree(node, depth) -> int:
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def get_const_size(sent, label) -> tuple:
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


def compute_constituents_metrics(sent) -> dict:
    constituents = {'S_len': len(sent), 'Depth': walk_tree(sent.root,1)}

    constituents['NP_max'], constituents['NP_avg'] = get_const_size(sent, "NP")
    constituents['PP_max'], constituents['PP_avg'] = get_const_size(sent, "PP")
    constituents['ADVP_max'], constituents['ADVP_avg'] = get_const_size(sent, "ADVP")
    constituents['ADJP_max'], constituents['ADJP_avg'] = get_const_size(sent, "ADJP")

    return constituents


def np_val(x) -> int:
    (a) = x._.labels
    try:
        label = a[0]
    except IndexError:
        label = "None"
    if label == "NP":
        return 1
    else:
        return 0



def compute_parse_tree_metrics(sent) -> dict:
    head_final_count = 0
    head_first_count = 0
    count_final_NP = 0
    count_first_NP = 0
    for token in sent:
        if token.i < token.head.i:
            head_final_count += 1
            if token.tag_ == "NN":
                count_final_NP += 1
        elif token.i > token.head.i:
            head_first_count += 1
            if token.tag_ == "NN":
                count_first_NP += 1
        else:
            pass
    parse_tree = {'right_branches_all': head_first_count/len(sent),
                  'left_branches_all': head_final_count/len(sent),
                  'right_branches_NP': count_first_NP / len(sent),
                  'left_branches_NP': count_final_NP / len(sent),
                  'branching_index_all': head_final_count - head_first_count}

    return parse_tree

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



def make_bert_datasets(df: pd.DataFrame) -> list:
    df = format(df)
    df = df.drop(['Original'], axis = 1)
    df.columns = ['labels', 'text']
    df['labels'] = df['labels'].map({'Automated':1, "Human":0})
    return df

# if __name__ == "__main__":
#     nlp = spacy.load('en_core_web_md')
#     if spacy.__version__.startswith('2'):
#         nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
#     else:
#         nlp.add_pipe("benepar", config={"model": "benepar_en3"})

#     doc = nlp("The time for action is now.")
#     doc = nlp("Every man asked some actress that he met about some play that she appeared in.")
#     #doc = nlp("I throw the ball to the dog")

#     sent = list(doc.sents)[0]

#     for sent in doc.sents:
#         print(type(sent))
#         print(compute_parse_tree_metrics(sent))
    
    # print(sent._.parse_string)

    # print(compute_density_metrics(sent))
    # print(compute_constituents_metrics(sent))
    # print(compute_parse_tree_metrics(sent))

    # fix character when reading automated translations
    # sent.replace("&#39;", "'")