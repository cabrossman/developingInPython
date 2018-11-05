#https://www.learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/
#https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134
#https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html
"""
Without LSH - we would have to look at all cominations of nearest neighbors - too costly
LSH compresses rows into "signatures (sequences of integers) used to compare items without going into low granularity

LSH Goals
1. hash items such that similar items go into the same bucket with high probability
2. restrict similarity search to the bucket associated with the query items

Shingles : the lowest level unit
1. choose unique shingles so that the probability of a shingle appearing in a given docusment is low (the is not a good word for example - appears in all documents - think TFIDF)

Minhash Signatures
1. Replace the large set with a smaller "signature" that still preserves the underlying similarity metric
2. This is done by randomly ordering the rows and grabbing the index of the first non-zero entry in each colmn
3. step two is repeated k times... the original matrix is n*m --> new matrix is k*m (where k<<n)

The similarity of the signatures is the fraction of the min-hash functions (rows) in which they agree - Jaccard similarity 

Locaity-sensative hashing --> find docs with Jaccard similarity of atleast T
1. hash columns of signature matrix M using several hash functions
2. If 2 documents hash into the same bucket for atleast one of the hashes we can take these two as a candidate pair

Band partition:
1. divide the signature matrix into b bands --> each band having r rows
2. for each band, hash its portion of each column to haash table with k buckets
3. cndidate column pairs are those that hash into the same bucket for atleast one band
4. tune b & r to catch most similar pairs but few non-similar pairs

high b implies lower similarity threshold (higher false positives) and lower b implies higher siilarity threshold (higher false negatives)


example

-100k documents stored as signature of length 100
-Signature matrix: 100*100000
-Brute force comparison of signatures will result in 100C2 comparisons = 5 billion (quite a lot!)
-Let’s take b = 20 → r = 5

similarity threshold (t) : 80%

similarity = t = sim(C1,C2) --> 1-(1-t^r)^b
"""

%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns # makes the graph prettier

s1 = "The cat sat on the mat."
s2 = "The red cat sat on the mat."

similarities = []
for shingle_size in range(2, 6):
    shingles1 = set([s1[max(0, i - shingle_size):i] for i in range(shingle_size, len(s1) + 1)])
    shingles2 = set([s2[max(0, i - shingle_size):i] for i in range(shingle_size, len(s2) + 1)])
    jaccard = len(shingles1 & shingles2) / len(shingles1 | shingles2)
    similarities.append(jaccard)

_ = plt.bar([2,3,4,5], similarities, width=0.25)
_ = plt.xlabel('Jaccard Similarity')
_ = plt.ylabel('Shingle Size')

#Impact of shingle size to similarity of documents. The larger the character shingles are the lower the similarity values tend to be.

import itertools

# from lsh import lsh, minhash # https://github.com/mattilyra/lsh

# a pure python shingling function that will be used in comparing
# LSH to true Jaccard similarities
def get_shingles(text, char_ngram=5):
    """Create a set of overlapping character n-grams.
    
    Only full length character n-grams are created, that is the first character
    n-gram is the first `char_ngram` characters from text, no padding is applied.

    Each n-gram is spaced exactly one character apart.

    Parameters
    ----------

    text: str
        The string from which the character n-grams are created.

    char_ngram: int (default 5)
        Length of each character n-gram.
    """
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))


def jaccard(set_a, set_b):
    """Jaccard similarity of two sets.
    
    The Jaccard similarity is defined as the size of the intersection divided by
    the size of the union of the two sets.

    Parameters
    ---------
    set_a: set
        Set of arbitrary objects.

    set_b: set
        Set of arbitrary objects.
    """
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)