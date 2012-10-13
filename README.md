tesla-celab
===========

This is the implementation of the TESLA-CELAB machine translation evaluation
metric as describe in the ACL 2012 paper.

    Character-Level Machine Translation Evaluation for Languages with Ambiguous Word Boundaries. Chang Liu and Hwee Tou Ng. 2012. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics.

Contents
--------

- `scripts/`: the implementation is a single file tc-match.py
- `resources/`: the CILIN Chinese synonym dictionary. Required by tc-match.py.
- `testdata/`: test data and sample usage.

For an example use, go to `testdata`, run `example.sh` and the output should
match `scores-output` exactly. The input files are Chinese texts where
characters are separated by single spaces. The reason the separation is not
done automatically is that even in character-based Chinese segmentation, one
segment is not always equivalent to one UTF character. For example, in the
following sentence, '500' could be considered a single segment depending on
your objectives.

    我 想 要 一 把 500 日 元 的 扇 子 。

The output file contains one line for each input sentence, containing the
TESLA-CELAB score measuring the semantic similarity between the reference
sentence and the system-generated translation. The score is between 0 and 1.

All files are encoded in UTF-8.

Dependencies
------------

TESLA-CELAB depends on cvxopt to solve the linear programming problem. You can
download cvxopt at http://abel.ee.ucla.edu/cvxopt/.

Contact
-------

If you use TESLA-CELAB in your research, please cite the ACL 2012 paper
mentioned above. Suggestions and comments are most welcome; please send them to
liuchangjohn@gmail.com.
