import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # implement the recognizer
    for index in range(test_set.num_items):

        top_prob, top_word = float("-inf"), None

        word_probabilities = {}

        seq, lengths = test_set.get_item_Xlengths(index)
        for word, model in models.items():
            try:
                word_probabilities[word] = model.score(seq, lengths)
            except Exception as e:
                word_probabilities[word] = float("-inf")

            if word_probabilities[word] > top_prob:
                top_prob, top_word = word_probabilities[word], word

        probabilities.append(word_probabilities)
        guesses.append(top_word)

    return probabilities, guesses