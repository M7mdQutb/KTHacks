import numpy as np

class Lang_Dict():
    def __init__(self):
        self.data = {}

    def addToDict(self,letter: str,hand_pos: np.array, chest_pos: np.array) -> str:
        """
            Adds a new letter to the dictionary using given positions.
        """
        if letter in self.data:
            return f"Letter '{letter}' already exists in the dictionary."
        
        try:
            condition = np.array([])
            self.data[tuple(condition)] = letter
            return True
        except Exception as e:
            return f"Error that idk about: {e}"


def translate(hand_pos: np.array,pose_pos: np.array, lang_dict: Lang_Dict) -> str:
    """
        Translates a letter using the provided language dictionary.
    """
    try:
        wristPos = hand_pos[0]
        thumbPos = hand_pos[1]
        indexPos = hand_pos[2]
        middlePos = hand_pos[3]
        ringPos = hand_pos[4]
        pinkyPos = hand_pos[5]

        condition = np.array([
            
        ])
        if tuple(condition) in lang_dict.data:
            return lang_dict.data[tuple(condition)]
        else:
            raise ValueError("Value not found in the language dictionary.")
    except Exception as e:
        raise e