#!/bin/bash
set -e
set -x

../scripts/tc-match.py --cilin ../resources/cilin.utf8 ref.txt sys-tch.txt scores-output
