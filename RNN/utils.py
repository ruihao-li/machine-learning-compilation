import io
import os
import glob
import unicodedata
import string
import torch
import random

# All alphabets (small + capital letters) and ".,;'"
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# print(unicode_to_ascii('Ślusàrski'))

def load_data():
    """
    Build the category_lines dictionary, a list of names per language.
    """
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)

    def read_lines(filename):
        """
        Read a file and split into lines.
        """
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories

# print(load_data()[0]['Vietnamese'][:10])


######################################################################
"""
To represent a single letter, we use a "one-hot vector" of size <1 x n_letters>.
A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 ...>.
To make a word we join a bunch of those into a 2d tensor <line_length x 1 x n_letters>.
The extra 1 dimension is for the batch dimension -- we are just using a batch size of 1 here.
"""

def letter_to_index(letter):
    """
    Turn a letter into an index.
    """
    return all_letters.find(letter)

def letter_to_tensor(letter):
    """
    Turn a letter into a <1 x n_letters> tensor.
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    """
    Turn a line into a <line_length x 1 x n_letters> tensor.
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for line_index, letter in enumerate(line):
        tensor[line_index][0][letter_to_index(letter)] = 1
    return tensor

# print(letter_to_tensor('J'))
# print(line_to_tensor('Jones').size())

######################################################################

def category_from_output(output, all_categories):
    """
    Get the index of the greatest value in the output.
    """
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def random_choice(l):
    """
    Choose a random element from a list.
    """
    return l[random.randint(0, len(l) - 1)]