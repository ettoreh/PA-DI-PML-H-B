import random
import numpy as np



def get_random_watermark(T: int) -> np.array:
    """_summary_

    Args:
        T (int): size of the watermark we want to embbed 

    Returns:
        np.array: array of T value either 0 or 1
    """
    return np.array(random.choices([0,1], k=T))

def get_watermark_from_text(text: str) -> np.array:
    """_summary_

    Args:
        text (str): a text to embbed

    Returns:
        np.array: array of the text into binary representation
    """
    binary_str = ''.join(format(ord(c), '08b') for c in text)
    return np.array([int(b) for b in binary_str])

def get_text_from_watermark(watermark: np.array) -> str:
    """_summary_

    Args:
        watermark (np.array): array of either 0 or 1

    Returns:
        str: the embbeded text
    """
    character = [str(bit) for bit in watermark]
    character = [character[i*8:(i+1)*8] for i in range(len(character)//8)]
    character = [chr(int(''.join(char), 2)) for char in character]
    return ''.join(character)
