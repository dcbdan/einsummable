#!/usr/bin/env python3

import sys
import numpy as np

# Read from standard input and write to standard output
for line in sys.stdin:
    print("Successfully read from python: ", line)
    # Randomly generate 147 integers in the range [0, 3]
    random_integers = np.random.randint(0, 4, size=147)

    # Convert the array to a string of numbers separated by commas
    output_string = ",".join(map(str, random_integers))
    # Write to standard output
    sys.stdout.write(output_string)

    # Flush stdout to ensure it's written immediately
    sys.stdout.flush()
