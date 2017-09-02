from my_model_selectors import SelectorCV, SelectorBIC, SelectorDIC
import numpy as np
import pandas as pd
from asl_data import AslDb
import timeit

asl = AslDb() # initializes the database
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']
training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
for word in words_to_train:
    #try:
    start = timeit.default_timer()
    model = SelectorDIC(sequences, Xlengths, word,
                        min_n_components=2, max_n_components=15, random_state=14).select()
    end = timeit.default_timer()-start
    #except ValueError as ve:
    #    model=None
    #    print('caught an valueerror for word {}'.format(word))
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds (BIC {})"
              .format(word, model.n_components, end, getattr(model, 'dic', None)))
    else:
        print("Training failed for {}".format(word))

for word in words_to_train:
    #try:
    start = timeit.default_timer()
    model = SelectorCV(sequences, Xlengths, word,
                        min_n_components=2, max_n_components=15, random_state=14).select()
    end = timeit.default_timer()-start
    #except ValueError as ve:
    #    model=None
    #    print('caught an valueerror for word {}'.format(word))
    if model is not None:
        print("Training complete for {} with {} states with time {} seconds (CV {})"
              .format(word, model.n_components, end, getattr(model, 'cvs', None)))
    else:
        print("Training failed for {}".format(word))