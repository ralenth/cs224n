"""Practical 1

Greatly inspired by Stanford CS224 2019 class.
"""

import sys

import pprint

import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import random
import nltk

nltk.download('reuters')
nltk.download('pl196x')
import random

import numpy as np
import scipy as sp
from nltk.corpus import reuters
from nltk.corpus.reader import pl196x
from sklearn.decomposition import PCA, TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


#################################
# TODO: a)
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the 
            corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the 
            corpus
    """
    corpus_words = [word for sentence in corpus for word in sentence]
    corpus_words = sorted(set(corpus_words))
    num_corpus_words = len(corpus_words)

    return corpus_words, num_corpus_words


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]     
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted(list(set([
    'Ala', 'END', 'START', 'i', 'kot', 'lubic', 'miec', 'pies'])))
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)


#################################
# TODO: b)
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window.
            Words near edges will have a smaller number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the 
                same as the ordering of the words given by the distinct_words 
                function.
            word2Ind (dict): dictionary that maps word to index 
                (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = {word: i for i, word in enumerate(words)}

    for sentence in corpus:
        for i, word in enumerate(sentence):
            tokens = sentence[max(i - window_size, 0): i + 1 + window_size]
            for token in tokens:
                if token != word:
                    M[word2Ind[word], word2Ind[token]] += 1

    return M, word2Ind

# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]     
M_test, word2Ind_test = compute_co_occurrence_matrix(
    test_corpus, window_size=1)

# Correct M and word2Ind
M_test_ans = np.array([
    [0., 0., 2., 0., 0., 1., 1., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [2., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [0., 1., 0., 1., 0., 1., 1., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 1., 0., 0., 0., 0.]
])

word2Ind_ans = {
    'Ala': 0, 'END': 1, 'START': 2, 'i': 3, 'kot': 4, 'lubic': 5, 'miec': 6,
    'pies': 7}

# Test correct word2Ind
assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans, word2Ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2Ind_ans.keys():
    idx1 = word2Ind_ans[w1]
    for w2 in word2Ind_ans.keys():
        idx2 = word2Ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)


#################################
# TODO: c)
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality
        (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following
         SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number 
                of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)):
            matrix of k-dimensioal word embeddings.
            In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None

    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.fit_transform(M)
    print("Done.")

    return M_reduced

# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness 
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]  
M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 8), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 8)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)


#################################
# TODO: d)
def plot_embeddings(M_reduced, word2Ind, words, filename=None):
    """ Plot in a scatterplot the embeddings of the words specified 
        in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the
            corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to
            visualize
            filename (str): if not None, embedding plot will be saved to a file
            of given name
    """
    # choose only rows (tokens) given by "words" parameter
    # print(words, M_reduced.shape, [word2Ind[word] for word in words])
    M_chosen = M_reduced[np.array([word2Ind[word] for word in words]), :]

    fig, ax = plt.subplots()
    plt.title('2-dimensional embeddings of word vectors')
    ax.plot(M_chosen[:, 0], M_chosen[:, 1], 'o')
    for word in words:
        ax.annotate(
            word, (M_reduced[word2Ind[word], 0], M_reduced[word2Ind[word], 1])
        )

    if filename is not None:
        plt.savefig(filename)
    plt.show()


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# The plot produced should look like the "test solution plot" depicted below. 
# ---------------------

print ("-" * 80)
print ("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2Ind_plot_test = {
    'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words, 'test_embeddings.png')

print ("-" * 80)


#################################
# TODO: e)
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------

def read_corpus_pl():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    pl196x_dir = nltk.data.find('corpora/pl196x')
    pl = pl196x.Pl196xCorpusReader(
        pl196x_dir, r'.*\.xml', textids='textids.txt', cat_file="cats.txt")
    tsents = pl.tagged_sents(fileids=pl.fileids(), categories='cats.txt')[:5000]

    return [[START_TOKEN] + [
        w[0].lower() for w in list(sent)] + [END_TOKEN] for sent in tsents]


def plot_unnormalized(corpus, words, filename=None):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    plot_embeddings(M_reduced_co_occurrence, word2Ind_co_occurrence, words, filename)


def plot_normalized(corpus, words, filename=None):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, filename)

pl_corpus = read_corpus_pl()
words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]

plot_normalized(pl_corpus, words, 'normalized_embeddings.png')
plot_unnormalized(pl_corpus, words, 'unnormalized_embeddings.png')

#################################
# Co-occurrence plot analysis

# Q: What clusters together in 2-dimensional embedding space?
# A: Without normalization we observe three clusters: 1) śpiewaczka, poeta,
# obywatel, 2) literatura, 3) sztuka. With normalization we observe only
# two clusters: 1) śpiewaczka, poeta, obywatel, literatura, 2) sztuka.

# Q: What doesn’t cluster together that you might think should have?
# A: On both plots (with or without normalization) sztuka and literatura
# are placed far away from each other, which is obviously wrong, since
# literature (literatura) is a subfield of art (sztuka).

# Q: TruncatedSVD returns U × S, so we normalize the returned vectors in the second
# plot, so that all the vectors will appear around the unit circle. Is normalization
# necessary?
# A: In my opinion not only normalization was unnecessary, it even resulted in
# worse visualization, but I still think that normalization could help, if it
# was done before SVD.
#################################


#################################
# Section 2:
#################################
# Then run the following to load the word2vec vectors into memory. 
# Note: This might take several minutes.
wv_from_bin_pl = KeyedVectors.load("word2vec_100_3_polish.bin")


# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This may take several minutes
# -----------------------------------


#################################
# TODO: a)
def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors
                         loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    words = list(wv_from_bin.key_to_index.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

# -----------------------------------------------------------------
# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions
# Note: This may take several minutes
# -----------------------------------------------------------------

#################################
# TODO: a)
M, word2Ind = get_matrix_of_vectors(wv_from_bin_pl, words)
M_reduced = reduce_to_k_dim(M, k=2)

words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]
plot_embeddings(M_reduced, word2Ind, words, 'word2vec_embeddings.png')


#################################
# Reducing dimensionality of Word2Vec Word Embeddings

# Q: What clusters together in 2-dimensional embedding space?
# A: We can observe four clusters: 1) sztuka, 2) śpiewaczka, 3) poeta,
# 4) literatura, obywatel. It's worth noticing that we could draw a line,
# that would separate more abstract concepts (sztuka, literatura) from
# professions (obywatel, poeta, śpiewaczka).

# Q: What doesn’t cluster together that you might think should have?
# A: "Literatura" and "sztuka" are still too far away from each other in my
# opinion, especially that currently "obywatel" is closest word to "literatura".

# Q: How is the plot different from the one generated earlier from the
# co-occurrence matrix?
# A: First of all, "śpiewaczka" and "poeta" were almost indistinguishable in the
# 2D embedding of co-occurence matrix, which is not the case in word2vec embeddings.
# That could mean, that with word2vec we can distinguish between different subfields
# of art, which is a good thing. Secondly, "obywatel" is not so close to artistic
# professions and "sztuka" and "literatura" are closer to each other, which is both
# a good thing.
#################################


#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.

words = ['język', 'zamek', 'narcyz', 'ślimak', 'strzemiączko', 'fala', 'babka',
         'gałąź', 'nurt', 'komórkowy']

for word in words:
    polysemous = wv_from_bin_pl.most_similar(word)
    print(word, [polysemous[i][0] for i in range(10)])


#################################
# Words I tried unsuccessfully:

# język ['jeżyk', 'dialekt', 'słownictwo', 'polszczyzna', 'idiom', 'alfabet', 'narzecze', 'zyku', 'gramatyka', 'wjęzyku']
# zamek ['zameczek', 'pałac', 'zamczysko', 'forteca', 'grodź', 'warownia', 'gród', 'donżon', 'dworzyszcze', 'cytadela']
# narcyz ['hiacynt', 'żmichowska', 'róża', 'żmichowskiej', 'niezapominajka', 'anieli', 'narcissus', 'fiołek', 'irys', 'turchan']
# ślimak ['małż', 'dżdżownica', 'krab', 'węgorz', 'robak', 'skorupiak', 'żółwi', 'larwa', 'winniczek', 'kraby']
# strzemiączko ['kowadełko', 'przyssawka', 'obrąbka', 'przylga', 'parapodium', 'przylgowy', 'stapes', 'siedzeniowy', 'brzusiec', 'ambulakralny']
# fala ['kipiel', 'prąd', 'przybój', 'strumień', 'wir', 'kaskada', 'chmura', 'nawałnica', 'podmuch', 'burza']
# babka ['babcia', 'ciotka', 'wnuczka', 'siostra', 'teściowa', 'matka', 'kuzynka', 'ciocia', 'babki', 'synowa']
# gałąź ['gałęzie', 'konar', 'gałązka', 'gałęź', 'pień', 'łodyga', 'pnącze', 'korzeń', 'łodyżka', 'liść']

# Q: Why do you think many of the polysemous words you tried didn’t work?
# A: First of all, I think that some of these words would work, if we didn't limit
# ourselves to only top 10 words; for example I think "ślimak" is usually used to
# describe an animal, not a part of human ear, so its top 10 word are reasonable.
# Secondly, I think dataset, on which these were trained, could be to blame. For
# example "narcissus" is not a polish word and some others seem outdated, such as
# "kipiel" or "donżon".

# And successfully:
# nurt ['nurty', 'prąd', 'nurtowy', 'kręg', 'wątek', 'meander', 'odłam', 'zakolać', 'strumień', 'koryto']
# Word "nurt" has two basic meanings: 1) water stream, 2) trend, fraction. While almost all
# most similar words are close to the first meaning, "odłam" refers to the latter one.

# komórkowy ['receptorowy', 'błonowy', 'neurotransmiter', 'acetylocholina', 'mitochondrium', 'przewodowy', 'organellum', 'neuronowy', 'bezprzewodowy', 'neurohormon']
# Word "komórkowy" is usually associated with two things: 1) biological cell, 2) telephone.
# While words such as "receptorowy" or "organellum" are referring to first meaning, "przewodowy"
# and "bezprzewodowy" are clearly related to the second one.
#################################


#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.

w1 = "radosny"
w2 = "pogodny"
w3 = "smutny"
w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
w1_w3_dist = wv_from_bin_pl.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

# Words I tried:
# Synonyms ładny, piękny have cosine distance: 0.2338840365409851
# Antonyms ładny, brzydki have cosine distance: 0.3028407096862793
# Synonyms wielki, ogromny have cosine distance: 0.21717435121536255
# Antonyms wielki, mały have cosine distance: 0.3387768864631653
# Synonyms błyskotliwy, inteligentny have cosine distance: 0.3181685209274292
# Antonyms błyskotliwy, głupi have cosine distance: 0.6000652313232422
# Synonyms spokojny, opanowany have cosine distance: 0.48047226667404175
# Antonyms spokojny, gwałtowny have cosine distance: 0.7655699402093887

w1 = "ciekawy"
w2 = "porywający"
w3 = "nudny"
w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
w1_w3_dist = wv_from_bin_pl.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))
# Synonyms ciekawy, porywający have cosine distance: 0.6683554649353027
# Antonyms ciekawy, nudny have cosine distance: 0.5770048797130585

# Q: Once you have found your example, please give a possible explanation for why
# this counter-intuitive result may have happened.
# A: Machine learning models learn from data. In this case, we have thousand and
# thousands of sentences, where each one provides us with some context. It's
# reasonable for antonyms to be used in similar context, what could result in
# smaller cosine distance than expected. Moreover, words "ciekawy" and "nudny"
# are probably used much more often than "porywający" (which is more
# sophisticated).


#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------

# ------------------
# Write your analogy exploration code here.
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["syn", "kobieta"], negative=["mezczyzna"]))

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["ojciec", "kobieta"], negative=["mezczyzna"]))

# Q: In your solution please state the full analogy in the form x:y :: a:b
# A: mężczyzna:ojciec :: kobieta:matka
# [('matka', 0.7249778509140015),
#  ('dziecko', 0.6641475558280945),
#  ('żona', 0.6380440592765808),
#  ('siostra', 0.6344905495643616),
#  ('rodzic', 0.6329774856567383),
#  ('mąż', 0.6319166421890259),
#  ('córka', 0.6132293939590454),
#  ('dziewczyna', 0.6047946810722351),
#  ('chłopiec', 0.6006911993026733),
#  ('dziewczę', 0.5869620442390442)]


#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["naukowiec", "kobieta"], negative=["mezczyzna"]))

# A: mężczyzna:naukowiec :: kobieta:naukowczyni
# [('badacz', 0.6311038732528687),
#  ('osoba', 0.6220046877861023),
#  ('ludzie', 0.5903834104537964),
#  ('psycholog', 0.5849590301513672),
#  ('intelektualista', 0.5756651163101196),
#  ('antropolog', 0.5753723978996277),
#  ('tubylec', 0.5661820769309998),
#  ('amerykan', 0.5496345162391663),
#  ('socjolog', 0.545727550983429),
#  ('biolog', 0.5391459465026855)]
# There's zero words here in female forms.


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and 
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'szef'], negative=['mezczyzna']))

# Q: # Which terms are most similar to ”kobieta” and ”szef” and most dissimilar
# to ”mezczyzna”?
# A: Honestly, we didn't get any words in female form and I'd say all of them
# are more similar to "mezczyzna" than "kobieta", but according to word embeddings:
# [('własika', 0.5678122639656067),
#  ('agent', 0.5483713150024414),
#  ('oficer', 0.5411549210548401),
#  ('esperów', 0.5383270978927612),
#  ('interpol', 0.5367037653923035),
#  ('antyterrorystyczny', 0.5327680110931396),
#  ('komisarz', 0.5326411128044128),
#  ('europolu', 0.5274547338485718),
#  ('bnd', 0.5271410346031189),
#  ('pracownik', 0.5215375423431396)]


pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['mezczyzna', 'prezes'], negative=['kobieta']))
# Q: Which terms are most similar to ”mezczyzna” and ”prezes” and most
# dissimilar to "kobieta"?
# A: [('wiceprezes', 0.6396454572677612),
#  ('czlonkiem', 0.5929950475692749),
#  ('przewodniczący', 0.5746127963066101),
#  ('czlonek', 0.5648552179336548),
#  ('przewodniczacym', 0.5586849451065063),
#  ('wiceprzewodniczący', 0.5560489892959595),
#  ('obowiazków', 0.5549101233482361),
#  ('obowiazani', 0.5544129610061646),
#  ('dyrektor', 0.5513691306114197),
#  ('obowiazany', 0.5471130609512329)]


#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors 
# ------------------
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['geniusz', 'kobieta'], negative=['mezczyzna']))
# Analogy: mężczyzna:geniusz :: kobieta:geniuszka
# Top 10 words:
# [('talent', 0.6390880942344666),
#  ('inteligencja', 0.5782135128974915),
#  ('sztuka', 0.5767902135848999),
#  ('indywidualność', 0.5501953363418579),
#  ('fantazja', 0.541026771068573),
#  ('uzdolnienie', 0.5374085307121277),
#  ('inwencja', 0.5360860228538513),
#  ('zdolności', 0.5355521440505981),
#  ('wyobraźnia', 0.5225419402122498),
#  ('artysta', 0.5196794271469116)]
# In my analysis I found gender bias, which is clearly visible in languages
# with many gender-specific words such as Polish language. While in English
# many words are neutral, in Polish "neutral" version is in fact male.


#################################
# TODO: h)
# Q: What might be the cause of these biases in the word vectors?
# A: Data used for training. In case of Polish language gender bias
# is expected, since Polish is generally gender biased. Moreover,
# once again because of the data, word vectors could be more sensitive
# for more common used words such as "ciekawy" instead of "porywający".


#################################
# Section 3:
# English part
#################################
def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each length 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


wv_from_bin = load_word2vec()

#################################
# TODO: 
# Find English equivalent examples for points b) to g).


#################################
# TODO: b)
polysemous = wv_from_bin.most_similar('stream')  # nurt
print('stream', [polysemous[i][0] for i in range(10)])
# stream ['streams', 'streaming', 'indecipherable_chatter', 'Startup_mSpot_lets', 'Honopou', 'Overflowing_lakes', 'Harmattan_Co', 'bisects_bunker', 'extragalactic_origin', 'torrent']

polysemous = wv_from_bin.most_similar('cellular')    # komórkowy
print('cellular', [polysemous[i][0] for i in range(10)])
# cellular ['Dr._Andrei_Gudkov', 'phone_reverse_lookup', 'celluar', 'Femto_cells', 'wireless', 'ceramic_particulate_filter', 'telecommunication', 'cell_phone', 'GSM_cellular', 'cellphone']


#################################
# TODO: c)
w1 = "interesting"  # ciekawy
w2 = "thrilling"   # porywający
w3 = "boring"    # nudny
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))
# Synonyms interesting, thrilling have cosine distance: 0.5823081433773041
# Antonyms interesting, boring have cosine distance: 0.5772338211536407


#################################
# TODO: d)
pprint.pprint(wv_from_bin.most_similar(
    positive=["father", "woman"], negative=["man"]))

# Q: In your solution please state the full analogy in the form x:y :: a:b
# A: man:father :: woman:mother
# [('mother', 0.8462507128715515),
#  ('daughter', 0.7899606823921204),
#  ('husband', 0.7560455799102783),
#  ('son', 0.7279756665229797),
#  ('eldest_daughter', 0.7120417952537537),
#  ('niece', 0.7096832990646362),
#  ('aunt', 0.6960803866386414),
#  ('grandmother', 0.689734160900116),
#  ('sister', 0.6895190477371216),
#  ('daughters', 0.6731119155883789)]


#################################
# TODO: e)
pprint.pprint(wv_from_bin.most_similar(
    positive=["scientist", "woman"], negative=["man"]))
# Q: In your solution please state the full analogy in the form x:y :: a:b
# A: man:scientist :: woman:scientist
# [('researcher', 0.7213959097862244),
#  ('biologist', 0.5944804549217224),
#  ('geneticist', 0.593985378742218),
#  ('microbiologist', 0.5772261619567871),
#  ('professor', 0.5715740323066711),
#  ('biochemist', 0.568531334400177),
#  ('physicist', 0.561724841594696),
#  ('Researcher', 0.5584948062896729),
#  ('anthropologist', 0.5538322329521179),
#  ('molecular_biologist', 0.5461257100105286)]


#################################
# TODO: f)
pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'boss'], negative=['man']))
# Q: # Which terms are most similar to ”woman” and ”boss” and most dissimilar
# to ”man”?
# [('bosses', 0.5522644519805908),
#  ('manageress', 0.49151360988616943),
#  ('exec', 0.45940810441970825),
#  ('Manageress', 0.4559843838214874),
#  ('receptionist', 0.4474116563796997),
#  ('Jane_Danson', 0.44480547308921814),
#  ('Fiz_Jennie_McAlpine', 0.4427576959133148),
#  ('Coronation_Street_actress', 0.44275563955307007),
#  ('supremo', 0.4409853219985962),
#  ('coworker', 0.43986251950263977)]

pprint.pprint(wv_from_bin.most_similar(
    positive=['man', 'chairman'], negative=['woman']))
# Q: Which terms are most similar to ”man” and ”chairman” and most
# dissimilar to "woman"?
# [('Chairman', 0.7163518667221069),
#  ('chariman', 0.6484905481338501),
#  ('chairmain', 0.6070235371589661),
#  ('chief_executive', 0.5947784781455994),
#  ('Chariman', 0.5882598757743835),
#  ('chaiman', 0.5767490267753601),
#  ('vicechairman', 0.5463533997535706),
#  ('chairman_emeritus', 0.5410268306732178),
#  ('cochairman', 0.5251522660255432),
#  ('supremo', 0.5208278298377991)]


#################################
# TODO: g)
pprint.pprint(wv_from_bin.most_similar(
    positive=['genius', 'woman'], negative=['man']))
# Analogy: man:genius :: woman:genius
# [('genuis', 0.5182446241378784),
#  ('muse', 0.5045610070228577),
#  ('brilliance', 0.5017262101173401),
#  ('geniuses', 0.4958648681640625),
#  ('artistry', 0.47173988819122314),
#  ('savant', 0.460582435131073),
#  ('fauvism', 0.4523790180683136),
#  ('reinvents_herself', 0.4516585171222687),
#  ('curvacious', 0.4485305845737457),
#  ('Gabrielle_Chanel', 0.4475356340408325)]


#################################
# TODO: i)
# Q: Load vectors for English and run similar analysis for points from b) to g).
# Have you observed any qualitative differences? Answer with up to 7 sentences.
# A: Generally results are similar, but English word vectors are much less gender
# biased. For example in e) when asked for a female scientist, Polish vectors
# resulted in words not connected to research such as "ludzie" or "amerykan",
# and all words connected to research were in male form ("badacz", "antropolog"),
# while English vectors returned only research-oriented words. In f1) when asked
# for a female boss, Polish vectors returned none words related to being in charge,
# English vectors were better at that, however we can still see some gender bias,
# since we got "receptionist". That being said, we still observe some bias in
# English vectors: second most similar word to a female genius is "muse", which
# is often used as "(female) inspiration for (male) geniuses". Another thing
# worth noticing is that word "chairman" is clearly biased. To sum up, English
# vectors allow us to obtain better word representations than Polish ones, what
# could be caused by data used for training and characteristics of these
# languages themselves.
