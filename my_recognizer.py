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
    # TODO implement the recognizer
    # return probabilities, guesses
    probabilities = []
    guesses = []
    X_lengths = test_set.get_all_Xlengths()
    for X, lengths in X_lengths.values():
        max_score = float("-inf") 
        best_guess = None 
        log_l = {}
        for word, model in models.items():
            try:
                word_score = model.score(X, lengths)
                log_l[word] = word_score

                if word_score > max_score:
                    max_score = word_score
                    best_guess = word
            except:
                # Unable to process word
                log_l[word] = float("-inf")

        guesses.append(best_guess)
        probabilities.append(log_l)

    return probabilities, guesses
