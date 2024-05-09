"""Generate Python lists in different file formats."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from json import dump as jdump
from pickle import dump

# %% File generation

# Generate an example Python list
data = list(range(1, 11))

# Generate a JSON file
with open('list.json', 'w') as file:
    jdump(data, file)

# Generate a Python binary file
with open('list.p', 'wb') as file:
    dump(data, file)

# Generate a text file
with open('list.txt', 'w') as file:
    for value in data:
        file.write(f"{value}\n")
