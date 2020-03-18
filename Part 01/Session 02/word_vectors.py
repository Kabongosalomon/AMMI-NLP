import io, sys
import numpy as np
from heapq import *

def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
    return data

## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    ## FILL CODE
    return u.dot(v)/(np.linalg.norm(u)* np.linalg.norm(v))

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search

def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = None

    ## FILL CODE
    for words in word_vectors.keys():
        temp = cosine(word_vectors[words], x)
        if temp > best_score and words not in exclude_words:
            best_score = temp
            best_word = words
            
    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []

    ## FILL CODE
    exclude = []    
    for ki in range(k+1):
        
        b_word = nearest_neighbor(x, vectors, exclude)
        score = cosine(x,  vectors[b_word])

        heap.append((score, b_word))
        exclude.append(b_word)

    return [heappop(heap) for i in range(len(heap))][::-1][:-1] # reverse and don't take the last element
## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    ## FILL CODE
    a = a.lower()
    b = b.lower()
    c = c.lower()
    
    best_anal = -np.inf
    best_anal_word = ''
    
    x_a = word_vectors[a]/np.linalg.norm(word_vectors[a])
    x_b = word_vectors[b]/np.linalg.norm(word_vectors[b])
    x_c = word_vectors[c]/np.linalg.norm(word_vectors[c])
    
    for word in word_vectors.keys():
        
        if True in [i in word for i in [a, b, c]]:
            continue
            
        word_vectors[word] = word_vectors[word]/ np.linalg.norm(word_vectors[word])
        
        anal = (x_c+x_b-x_a).dot(word_vectors[word])
        
        if anal > best_anal:
            best_anal = anal
            best_anal_word = word
        
    return best_anal_word

## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    strength = 0.0
    ## FILL CODE
    a_sum = 0.0
    b_sum = 0.0
    
    for a in A : 
        a_sum += cosine(vectors[w], vectors[a])
    
    for b in B : 
        b_sum += cosine(vectors[w], vectors[b])
    
    
    
    strength = 1/len(A) * a_sum - 1/len(B) * b_sum
    return strength

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    score = 0.0
    ## FILL CODE
    score_1 = 0.0
    score_2 = 0.0
    for w in X:
        score_1 += association_strength(w, A, B, vectors)
    
    for z in Y:
        score_2 += association_strength(z, A, B, vectors)
     
    score = score_1 - score_2
    return score

######## MAIN ########

print('')
print(' ** Word vectors ** ')
print('')

word_vectors = load_vectors(sys.argv[1])

print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))

print('')
print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print (word + '\t%.3f' % score)

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))

## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))

## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))
