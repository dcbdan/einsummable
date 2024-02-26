#!/usr/bin/env python3

import sys

# Read from standard input and write to standard output
for line in sys.stdin:
    print("Successfully read from python: ", line)
    # Write to standard output
    sys.stdout.write("hello from python.\n")

    # Flush stdout to ensure it's written immediately
    sys.stdout.flush()

