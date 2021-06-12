import benepar
from numpy import average
import spacy
from nltk import Tree, acyclic_breadth_first

#benepar.download('benepar_en3')

def compute_density_metrics(sent):
    densities = {'DET':0, 'AUX':0,'CONJ':0, 'PRON':0, 'ADP':0, 'PUNCT':0, 'FUNC':0}
    
    for w in sent:
        if w.pos_ in densities.keys():
            densities[w.pos_] += 1
        elif w.pos_ == 'SCONJ':
            densities['CONJ'] += 1
        elif w.pos_ == 'PART':
            densities['FUNC'] += 1
    
    densities['FUNC'] += (densities['DET'] + densities['AUX'] + densities['CONJ'] + densities['PRON'] + densities['ADP'] + densities['PUNCT'])
    cont_words = len(sent) - densities['FUNC']        
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


def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])

def to_nltk_tree(node, left, right):
    left = left + node.n_lefts
    right = right + node.n_rights
    print(node.text)
    print(left, right)
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child, left, right) for child in node.children])
    else:
        return tok_format(node)




def compute_constituents_metrics(sent):
    constituents = {'S_len': len(sent), 'Depth': walk_tree(sent.root,1)}
    constituents['NP_max'], constituents['NP_avg'] = get_const_size(sent, "NP")
    constituents['PP_max'], constituents['PP_avg'] = get_const_size(sent, "PP")
    constituents['ADVP_max'], constituents['ADVP_avg'] = get_const_size(sent, "ADVP")
    constituents['ADJP_max'], constituents['ADJP_avg'] = get_const_size(sent, "ADJP")
    return constituents

def compute_parse_tree_metrics(sent):
    constituent_iter = iter(sent._.constituents)
    count_left = 0
    count_right = 0
    for x in constituent_iter:
        if x._.parent != None:
            if list(x)[len(list(x)) - 1] == list(x._.parent)[len(list(x._.parent)) - 1]:
                count_right = count_right + 1
            if list(x)[0] == list(x._.parent)[0]:
                count_left = count_left + 1

    if count_right != 0:
        ratio = count_left/count_right
    else:
        ratio = 0

    parse_tree = {'right_branches': count_right, 'left_branches': count_left, 'ratio': ratio}
    return parse_tree


nlp = spacy.load('en_core_web_md')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

doc = nlp("The time for action is now.")
doc = nlp("Every man asked some actress that he met about some play that she appeared in")
#doc = nlp("I throw the ball to the dog")


sent = list(doc.sents)[0]


for sent in doc.sents:

# print(sent._.parse_string)



# print(compute_density_metrics(sent))
#print(compute_constituents_metrics(sent))
# print(compute_parse_tree_metrics(sent))




# fix character when reading automated translations
# sent.replace("&#39;", "'")