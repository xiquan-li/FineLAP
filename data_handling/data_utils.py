import torch
from re import sub
import numpy as np

def pad_sequence(data):
    """Pad sequences to the same length"""
    try: 
        if isinstance(data[0], np.ndarray):
            data = [torch.as_tensor(arr) for arr in data]
        padded_seq = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
        length = [x.shape[0] for x in data]
        return padded_seq, length
    except Exception as e:
        print(f"Error when padding sequence: {e}")
        for d in data: 
            if type(d) == torch.Tensor:
                print(f"Data: {d.shape}")
        raise e


def text_preprocess(sentence):

    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')
    return sentence
