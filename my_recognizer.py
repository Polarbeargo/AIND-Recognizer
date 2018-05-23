import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category = DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer
    # return probabilities, guesses
    for item in range(test_set.num_items):

        seq, length = test_set.get_item_Xlengths(item)
        prob_rec, best_guess, highest_prob = {}, None, None

        for word, model in models.items():
            try:
                prob_rec[word] = model.score(seq, length)
                if highest_prob < prob_rec[word] or highest_prob is None:
                   highest_prob, best_guess = prob_rec[word], word
            except:
                prob_rec[word] = None
                
        probabilities.append(prob_rec)
        guesses.append(best_guess)

    return probabilities, guesses
