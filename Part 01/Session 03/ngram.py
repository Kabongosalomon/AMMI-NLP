import io, sys, math, re
from collections import defaultdict
import numpy as np

# GOAL: build a stupid backoff ngram model

def load_data(filename):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    vocab = defaultdict(lambda:0)
    for line in fin:
        sentence = line.split()
        data.append(sentence)
        for word in sentence:
            vocab[word] += 1
    return data, vocab

def remove_rare_words(data, vocab, mincount=0):
    ## FILL CODE
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    for i1, words in enumerate(data):
        for i2, w in enumerate(words):
            if w not in vocab.keys():
                data[i1][i2] = '<unk>'
            elif vocab[w] < mincount:
                data[i1][i2] = '<unk>'
            
    return data


def build_ngram(data, n):
    total_number_words = 0
    counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    for sentence in data:
        sentence = tuple(sentence)
        ## FILL CODE
        # dict can be indexed by tuples
        # store in the same dict all the ngrams
        # by using the context as a key and the word as a value
        total_number_words += len(sentence)
        for gram in range(1, n+1):
            for i in range(len(sentence)-gram+1):
                counts[sentence[i:(gram+i-1)]][sentence[gram+i-1]]+=1

    prob  = defaultdict(lambda: defaultdict(lambda: 0.0))
    ## FILL CODE
    # Build the probabilities from the counts
    # Be careful with how you normalize!
    for context in counts.keys():
        for target in counts[context].keys():
            prob[context][target] = counts[context][target]/(sum(counts[context].values()))
    
    return prob

def get_prob(model, context, w):
    ## FILL CODE
    # code a recursive function over 
    # smaller and smaller context
    # to compute the backoff model
    # Bonus: You can also code an interpolation model this way
    if model[context][w] != 0:
        return model[context][w]
    else :
        return 0.4*get_prob(model, context[1:], w)

def perplexity(model, data, n):
    ## FILL CODE
    # Same as bigram.py
    perp = []
        
    for sentence in data:
        sentence = tuple(sentence)
        perp_s = 1.0
        for i in range(len(sentence)):
            if (n+i-1) < len(sentence):                
                perp_s *= 1.0/get_prob(model, sentence[i:(n+i-1)], sentence[n+i-1])
            
        perp.append(perp_s**(1/len(sentence)))
                
    return np.mean(perp)

def get_proba_distrib(model, context):
    ## FILL CODE
    # code a recursive function over context
    # to find the longest available ngram 
    if sum(model[context].values()) != 0:
        return context
    else :
        return get_proba_distrib(model, context[1:])

def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    sentence = ["<s>"]
    
    i = 0
    while True :
        
        avail_context = get_proba_distrib(model, \
                                        tuple(sentence[max(i+1-n, 0):i+1]))
#         ipdb.set_trace()
        a = list(model[avail_context].keys())
        prob = list(model[avail_context].values())
        current_pred = np.random.choice(a, 1, \
                                     p = prob)
        sentence.append(current_pred[0])
        i+=1
        if current_pred[0] == '</s>':
            break

    return sentence

###### MAIN #######

n = 2

print("load training set")
train_data, vocab = load_data("train.txt")

## FILL CODE
# Same as bigram.py
train_data = remove_rare_words(train_data, vocab, mincount = 3)

print("build ngram model with n = ", n)
model = build_ngram(train_data, n)

print("load validation set")
valid_data, _ = load_data("valid.txt")
## FILL CODE
# Same as bigram.py
valid_data = remove_rare_words(valid_data, vocab, mincount = 3)

print("The perplexity is", perplexity(model, valid_data, n))

print("Generated sentence: ",generate(model))

