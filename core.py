import numpy as np
import pickle
import os

class Lang_Dict():
    def __init__(self):
        self.data = {}

    def addToDict(self,letter: str,condition: tuple) -> str:
        """
            Adds a new letter to the dictionary using given positions.
        """
        if not isinstance(condition, tuple):
            raise TypeError("Condition must be a tuple or numpy array.")
        
        if condition in self.data:
            raise KeyError(f"Letter '{letter}' already exists in the dictionary.")
        
        try:
            self.data[condition] = letter
            return True
        except Exception as e:
            raise e

    def loadFromFile(self, file_path: str) -> bool:
        """
            Loads the dictionary from a file.
        """
        if not os.path.exists(file_path):
            return False
        
        try:
            with open(file_path, 'rb') as file:
                self.data = pickle.load(file)
            return True
        except Exception as e:
            print(f"Error loading dictionary from file: {e}")
            return False
        
    def saveToFile(self, file_path: str) -> bool:
        """
            Saves the dictionary to a file.
        """
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(self.data, file)
            return True
        except Exception as e:
            print(f"Error saving dictionary to file: {e}")
            return False

def translate(condition: tuple, lang_dict: Lang_Dict) -> str:
    """
        Translates a letter using the provided language dictionary.
    """
    try:
        if tuple(condition) in lang_dict.data:
            return lang_dict.data[tuple(condition)]
        else:
#            raise ValueError("Value not found in the language dictionary.")
            return "value not found"
    except Exception as e:
        raise e
    

def add_to_dict(letter: str, condition: tuple,lang_dict: Lang_Dict) -> str:
    """
        Adds a new letter to the language dictionary.
    """
    try:
        lang_dict.addToDict(letter, condition)
        return lang_dict
    except Exception as e:
        raise e
        return False
