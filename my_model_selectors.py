import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from collections import defaultdict


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
        #try:
        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        if self.verbose:
            print("model created for {} with {} states".format(self.this_word, num_states))
        return hmm_model
        #except:
        #    if self.verbose:
        #        print("failure on {} with {} states".format(self.this_word, num_states))
        #    return None


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

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        def calc_bic(model, X, lengths, component_count):
            try:
                logL = model.score(X, lengths)
            except ValueError as ve:
                print('Value Error for {} components'.format(component_count))
                return float('inf')
            feature_count = X.shape[1]
            observation_count = len(lengths)
            starting_probability_count = component_count - 1
            transition_prob_count = component_count * (component_count - 1)
            gaussian_mean_count = component_count * feature_count
            gaussian_variance_count = component_count * feature_count
            parameter_count = starting_probability_count + transition_prob_count \
                              + gaussian_mean_count + gaussian_variance_count

            bic = -2 * logL + parameter_count * np.log(observation_count)
            model.bic = bic
            return bic


        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_list = [(self.base_model(_), _) for _ in range(self.min_n_components, self.max_n_components)]
        best_model, best_component_count = min(model_list, key=lambda _: calc_bic(_[0], self.X, self.lengths, _[1]))

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    lldict = None

    def create_lldict(self):
        lldict = defaultdict(dict)
        for component_count in range(self.min_n_components, self.max_n_components):
            for word in self.words:
                try:
                    wordX, wordLengths = self.hwords[word]
                    model = GaussianHMM(n_components=component_count, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(wordX, wordLengths)
                    logL = model.score(wordX, wordLengths)
                except ValueError as ve:
                    logL = 0
                if logL:
                    lldict[component_count][word] = logL
        return lldict

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        '''
        lp = likelyhood propability
        alp = average antilikelyhood propabilities
        reference_dictionary_anti_propabilities  = A reference to look up scores for anti-propabilities. Speeds up computational speed
        '''

        #building log-likelihood dictionary
        if not self.lldict:
            self.lldict = self.create_lldict()

        def calc_dic(lcdict, word):
            return lcdict[word] - sum([v for k,v in lcdict.items() if k != word]) / (len(lcdict) - 1)

        dic_scores = ((calc_dic(self.lldict[_], self.this_word), _)
                      for _ in range(self.min_n_components, self.max_n_components)
                      if self.this_word in self.lldict[_])

        max_dict_score, optimal_component_count = max(dic_scores, key=lambda _: _[1])

        best_model = self.base_model(int(optimal_component_count))
        best_model.dic = max_dict_score
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        def calc_cv(model, sequences):
            scores = list()
            max_splits = min(3, len(sequences))
            split_method = KFold(random_state=self.random_state, n_splits=max_splits)
            for cv_train_idx, cv_test_idx in split_method.split(sequences):
                training_data, training_lengths = combine_sequences(cv_train_idx, sequences)
                test_data, test_lengths = combine_sequences(cv_test_idx, sequences)
                try:
                    model.fit(training_data, training_lengths)
                    logL = model.score(test_data, test_lengths)
                except ValueError as ve:
                    continue
                scores.append(logL)
            cv_score = np.mean(scores) if scores else float('-inf')
            model.cvs = cv_score
            return -cv_score

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_list = [(self.base_model(_), _) for _ in range(self.min_n_components, self.max_n_components)]
        best_model, best_component = min(model_list, key=lambda _: calc_cv(_[0], self.sequences))
        return best_model if best_model is not None else self.base_model(self.n_constant)

