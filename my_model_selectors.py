import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(
                    self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(
                    self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def BIC_score(self, n):

        model = self.base_model(n)
        logN = np.log(len(self.X))
        d = model.n_features
        logL = model.score(self.X, self.lengths)
        p = n ** 2 + 2 * d * n - 1
        BIC = -2.0 * logL + p * logN

        return BIC, model

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_model = None
        best_score = float("Inf")

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                score, model = self.BIC_score(n)
                if score < best_score:
                    best_score, best_model = score, model
            except Exception as e:
                continue

        return best_model if best_model is not None else self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_num_components = None
        best_DIC_score = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:

                log_P_X_i = self.base_model(n).score(self.X, self.lengths)
                sum_log_P_X_all_but_i = 0.
                words = list(self.words.keys())
                M = len(words)
                words.remove(self.this_word)
                for word in words:
                    if word != self.this_word:
                        try:
                            model_selector_all_but_i = ModelSelector(
                                self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose)
                            sum_log_P_X_all_but_i += model_selector_all_but_i.base_model(n).score(
                                model_selector_all_but_i.X, model_selector_all_but_i.lengths)
                        except:
                            M = M - 1

                DIC = log_P_X_i - sum_log_P_X_all_but_i / (M - 1)

                if best_DIC_score is None or best_DIC_score < DIC:
                    best_DIC_score, best_num_components = DIC, n
            except:
                pass

        if best_num_components is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_num_components = None
        best_avg_likelihood_log = None

        for n in range(self.min_n_components, self.max_n_components + 1):

            count_logL = 0
            sum_logL = 0.

            try:
                split_method = KFold(n_splits=3)
                for cv_train, cv_test in split_method.split(self.sequences):
                    X, lengths = combine_sequences(cv_train, self.sequences)

                    try:
                        sum_logL += self.base_model(n).score(X, lengths)
                        count_logL += 1
                    except:
                        pass

                if count_logL > 0:
                    avg_logL = sum_logL / count_logL
                    if best_avg_likelihood_log is None or best_avg_likelihood_log < avg_logL:
                        best_avg_likelihood_log, best_num_components = avg_logL, n
            except:
                pass
        if best_num_components is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)
