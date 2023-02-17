import numpy as np
import random



def get_secret_matrix(type: str, size: tuple) -> np.array:
    """_summary_

    Args:
        type (str): type of matrix to generate 
        size (tuple): size of this matrix

    Returns:
        np.array: a matrix 
    """
    T, M = size
    if type == "direct":
        x = np.zeros(size)
        for i in range(T):
            j = random.choice(range(M))
            x[i][j] = 1
            
    if type == "diff":
        x = np.zeros(size)
        for i in range(T):
            j1, j2 = random.sample(range(M), 2)
            x[i][j1] = 1
            x[i][j2] = -1
    
    if type == "rand":
        x = np.random.uniform(size=size)
        
    return x

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
    return [int(b) for b in binary_str]