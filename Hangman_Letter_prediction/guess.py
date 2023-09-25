import numpy as np
import collections
import re

def guess(self,word):
    # word = "apple"
    # word = "a__le"

    clean_word = word[::2].replace("_",".")
    
    # find length of passed word
    len_word = len(clean_word)
    
    # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
    current_dictionary = self.current_dictionary
    new_dictionary = []
    
    # iterate through all of the words in the old plausible dictionary
    for dict_word in current_dictionary:
        # continue if the word is not of the appropriate length
        if len(dict_word) != len_word:
            continue
            
        # if dictionary word is a possible match then add it to the current dictionary
        if re.match(clean_word,dict_word):
            new_dictionary.append(dict_word)

    # new_dictionary is the key
    self.current_dictionary = new_dictionary

    



