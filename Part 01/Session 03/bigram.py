import io, sys, math, re
from collections import defaultdict
import numpy as np

# GOAL: build a stupid backoff bigram model

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


def build_bigram(data):
    unigram_counts = defaultdict(lambda:0)
    bigram_counts  = defaultdict(lambda: defaultdict(lambda: 0.0))
    total_number_words = 0

    ## FILL CODE
    # Store the unigram and bigram counts as well as the total 
    # number of words in the dataset
    for i1, sentence in enumerate(data):
        total_number_words += len(sentence)
        for i2, _ in enumerate(sentence):
            unigram_counts[data[i1][i2]] += 1
            if i2+1 < len(sentence):
                bigram_counts[data[i1][i2]][data[i1][i2+1]] += 1

    unigram_prob = defaultdict(lambda:0)
    bigram_prob = defaultdict(lambda: defaultdict(lambda: 0.0))

    ## FILL CODE
    # Build unigram and bigram probabilities from counts
    for word in unigram_counts.keys():
        unigram_prob[word] = unigram_counts[word]/total_number_words
    
    for i1, words in enumerate(data):
        for i2, w in enumerate(words):
            if i2+1 < len(words):
                # P(w_t/w_{t-1}) = P(i2+1/i2)
                bigram_prob[data[i1][i2]][data[i1][i2+1]] = \
                    bigram_counts[data[i1][i2]][data[i1][i2+1]]/unigram_counts[data[i1][i2]]

        
        
    return {'bigram': bigram_prob, 'unigram': unigram_prob}

def get_prob(model, w1, w2):
    assert model["unigram"][w2] != 0, "Out of Vocabulary word!"
    ## FILL CODE
    # Should return the probability of the bigram (w1w2) if it exists
    # Else it return the probility of unigram (w2) multiply by 0.4
    if model['bigram'][w1][w2] != 0:
        return model['bigram'][w1][w2]
    else :
        return model['unigram'][w2] * 0.4

def perplexity(model, data):
    ## FILL CODE
    # follow the formula in the slides
    # call the function get_prob to get P(w2 | w1)
    
    perp = []
    
    for i1, sentence in enumerate(data):
        perp_s = 1.0
        sentence = tuple(sentence)
        for i2, w in enumerate(sentence):
            if i2+1 < len(sentence):
                perp_s *= 1.0/get_prob(model, data[i1][i2], data[i1][i2+1])
        
        perp.append(perp_s**(1/len(sentence)))
                
    return np.mean(perp)

def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    i = 0
    while True :
        current_pred = np.random.choice(list(model['bigram'][sentence[i]].keys()), 1, \
                                         p = list(model['bigram'][sentence[i]].values()))
        sentence.append(current_pred[0])
        i+=1
        if current_pred[0] == '</s>':
            break
    return sentence

###### MAIN #######

print("load training set")
train_data, vocab = load_data("train2.txt")
## FILL CODE 
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# rare words with <unk> in the dataset
train_data = remove_rare_words(train_data, vocab, mincount = 10)

print("build bigram model")
model = build_bigram(train_data)

print("load validation set")
valid_data, _ = load_data("valid2.txt")
## FILL CODE 
# If you have a Out of Vocabulary error (OOV) 
# call the function "remove_rare_words" to replace 
# OOV with <unk> in the dataset
valid_data = remove_rare_words(valid_data, vocab, mincount = 10)

print("The perplexity is", perplexity(model, valid_data))

print("Generated sentence: ",generate(model))