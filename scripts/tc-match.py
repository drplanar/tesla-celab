#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sqlite3
import sys
from pprint import pprint
from codecs import open
from itertools import product, chain
from cvxopt import matrix, spmatrix, solvers
from getopt import gnu_getopt as getopt
import numpy

solvers.options['show_progress'] = False

# globals
conn = sqlite3.connect(':memory:')
# conn = sqlite3.connect('match.db')
reference = None
paraphrase = None 
outputfile = None
verbose = False
simmethod = 'synonym'
separator = '_'
cilin = {}
max_ngram = 4
fnum = 0.25


def load_cilin(cilin_fname):
    global cilin

    for line in open(cilin_fname, 'r', 'utf-8'):
        # each line is of the form
        # Di15A04= 名位 名分 排名分
        # the dictionary is thus,
        #   '名位' => set('Di15A04'),
        #   '名分' => set('Di15A04')
        # etc.
        tokens = line.split()
        for w in tokens[1:]:
            try:
                cilin[w].add(tokens[0])
            except KeyError:
                cilin[w] = set([tokens[0],])


def parse_opts():
    optlist, args = getopt(sys.argv, '', [
            'verbose', 
            'simmethod=', 
            'fnum=',
            'cilin='])

    global reference, paraphrase, outputfile
    _, reference, paraphrase, outputfile = args
    reference = open(reference, 'r', 'utf-8')
    paraphrase = open(paraphrase, 'r', 'utf-8')
    outputfile = open(outputfile, 'w', 'utf-8')
    for opt, val in optlist:
        if opt == '--verbose':
            global verbose
            verbose = True
        elif opt == '--simmethod':
            global simmethod
            simmethod = val
        elif opt == '--cilin':
            load_cilin(val)
        elif opt == '--fnum':
            global fnum
            fnum = float(val)
        else:
            assert False


def iterate_ngrams(length):
    for start in range(length):
        for end in range(start+1, min(start+1+max_ngram, length+1)):
            yield (start, end)


def concat_ngram(char_pos_s, start, end):
    return ''.join([char_pos[0] for char_pos in char_pos_s[start : end]])


def unit_similarity(word1, word2):
    assert simmethod == 'synonym'
    if word1 == word2:
        return 1.0
    
    try:
        categories1 = cilin[word1]
        categories2 = cilin[word2]
        return 0.0 if categories1.isdisjoint(categories2) else 1.0
    except KeyError:
        return 0.0


def unit_similarities(ref_char_pos_s, para_char_pos_s):
    """Similarity links.

    Iterates over (ref-start, ref-end, para-start, para-end, similarity).

    """
    ref_length, para_length = len(ref_char_pos_s), len(para_char_pos_s)
    for (ref_start, ref_end), (para_start, para_end) in product(iterate_ngrams(ref_length), iterate_ngrams(para_length)):
        ref_ngram = concat_ngram(ref_char_pos_s, ref_start, ref_end)
        para_ngram = concat_ngram(para_char_pos_s, para_start, para_end)
        yield ref_start, ref_end, para_start, para_end, unit_similarity(ref_ngram, para_ngram)


def link_similarity(ref_start, ref_end, para_start, para_end, us):
    """us is (ref_start, ref_end, para_start, para_end) => similarity.

    This function uses dynamic programming.

    ref_idx in range [ref_start, ref_end)
    para_idx in range [para_start, para_end)

    Hence,

    ref_idx - ref_start in range [0, ref_end - ref_start)
    para_idx - para_start in range [0, para_end - para_start)

    m[ref_idx - ref_start][para_idx - para_start] is the best match score between
    - ref[ref_start : ref_idx] (both indices inclusive); and
    - para[para_start : para_idx] (both indices inclusive)

    """
    m = numpy.zeros((ref_end - ref_start, para_end - para_start), dtype=float)

    # terminal cases
    for ref_idx in range(ref_start, ref_end):
        try:
            m[ref_idx - ref_start][0] = us[(ref_start, ref_idx + 1, para_start, para_start + 1)]
        except KeyError:
            pass # still zero

    for para_idx in range(para_start, para_end):
        try:
            m[0][para_idx - para_start] = us[(ref_start, ref_start + 1, para_start, para_idx + 1)]
        except KeyError:
            pass # still zero

    for ref_idx, para_idx in product(range(ref_start + 1, ref_end), range(para_start + 1, para_end)):
        for prev_ref_idx, prev_para_idx in product(range(ref_start, ref_idx), range(para_start, para_idx)):
            prev_match = m[prev_ref_idx - ref_start][prev_para_idx - para_start] 
            if prev_match < 0.01:
                continue

            try:
                this_match = us[(prev_ref_idx + 1, ref_idx + 1, prev_para_idx + 1, para_idx + 1)]
            except KeyError:
                continue

            # how to combine the scores? let's multiply them for now
            new_match = prev_match * this_match
            if m[ref_idx - ref_start][para_idx - para_start] < new_match:
                m[ref_idx - ref_start][para_idx - para_start] = new_match

    # return the final entry
    return m[ref_end - ref_start - 1][para_end - para_start - 1]


def set_up_sqlite():
    global conn
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS link_similarity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ref_start INTEGER NOT NULL,
                    ref_end INTEGER NOT NULL,
                    para_start INTEGER NOT NULL,
                    para_end INTEGER NOT NULL,
                    similarity FLOAT NOT NULL,
                    varid INTEGER,
                    UNIQUE (ref_start, ref_end, para_start, para_end)
                )""")
    c.execute("DELETE FROM link_similarity")

    c.execute("""CREATE TABLE IF NOT EXISTS ngram_match_weight (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ref_or_para CHAR(1) NOT NULL, -- 'r' or 'p'
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    varid INTEGER,
                    UNIQUE (ref_or_para, start, end)
                )""")
    c.execute("DELETE FROM ngram_match_weight")

    c.execute("""CREATE TABLE IF NOT EXISTS covered_ngram_match_weight (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ref_or_para CHAR(1) NOT NULL, -- 'r' or 'p'
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    varid INTEGER,
                    UNIQUE (ref_or_para, start, end)
                )""")
    c.execute("DELETE FROM covered_ngram_match_weight")

    conn.commit()


def unit_link_similarity(ref_char_pos_s, para_char_pos_s):
    us = {}

    ref_length, para_length = len(ref_char_pos_s), len(para_char_pos_s)
    for ref_start, ref_end, para_start, para_end, similarity in unit_similarities(ref_char_pos_s, para_char_pos_s):
        if similarity < 0.01:
            continue
        
        us[(ref_start, ref_end, para_start, para_end)] = similarity

        if verbose:
            ref_ngram = concat_ngram(ref_char_pos_s, ref_start, ref_end)
            para_ngram = concat_ngram(para_char_pos_s, para_start, para_end)
            print 'unit similarity:', ref_ngram, para_ngram, similarity

    return us


def fill_in_link_similarity(ref_char_pos_s, para_char_pos_s, us):
    global conn
    c = conn.cursor()
    ref_length, para_length = len(ref_char_pos_s), len(para_char_pos_s)

    for (ref_start, ref_end), (para_start, para_end) in product(iterate_ngrams(ref_length), iterate_ngrams(para_length)):
        ls = link_similarity(ref_start, ref_end, para_start, para_end, us)
        if ls < 0.01:
            continue

        c.execute("""INSERT INTO link_similarity (ref_start, ref_end, para_start, para_end, similarity) 
                VALUES (?, ?, ?, ?, ?)""", (ref_start, ref_end, para_start, para_end, ls))

        if verbose:
            ref_ngram = concat_ngram(ref_char_pos_s, ref_start, ref_end)
            para_ngram = concat_ngram(para_char_pos_s, para_start, para_end)
            print 'compound similarity:', ref_ngram, para_ngram, ls

    conn.commit()


def fill_in_ngram_match_weight_variables():
    global conn
    c = conn.cursor()
    c.execute("""INSERT INTO ngram_match_weight (ref_or_para, start, end)
                 SELECT DISTINCT 'r', ref_start, ref_end FROM link_similarity""")
    c.execute("""INSERT INTO ngram_match_weight (ref_or_para, start, end)
                 SELECT DISTINCT 'p', para_start, para_end FROM link_similarity""")
    conn.commit()


def fill_in_covered_ngram_match_weight():
    global conn
    read = conn.cursor()
    read.execute("SELECT ref_or_para, start, end FROM ngram_match_weight")
    write = conn.cursor()
    for ref_or_para, parent_start, parent_end in read:
        for child_start in range(parent_start, parent_end):
            for child_end in range(child_start + 1, parent_end + 1):
                try:
                    write.execute("""
                            INSERT INTO 
                            covered_ngram_match_weight (id, ref_or_para, start, end)
                            VALUES ((SELECT MAX(id) + 1 FROM covered_ngram_match_weight), ?, ?, ?)
                            """, (ref_or_para, child_start, child_end))
                except sqlite3.IntegrityError:
                    # duplicates are ignored
                    pass
    conn.commit()


def reset_min_id(table_name, min_id):
    global conn
    c = conn.cursor()
    c.execute("SELECT MIN(id) FROM %s" % table_name)
    curren_min = c.fetchone()[0]
    if curren_min is None:
        return min_id
    delta = min_id - curren_min
    c.execute("UPDATE %s SET varid = id + %s" % (table_name, delta))
    conn.commit()
    c.execute("SELECT MAX(varid) + 1 FROM %s" % table_name)
    return c.fetchone()[0]


def normalize_variable_ids():
    global conn
    c = conn.cursor()
    var_idx = 0

    # the following will become variables:
    var_idx = reset_min_id("link_similarity", var_idx)
    var_idx = reset_min_id("ngram_match_weight", var_idx)
    var_idx = reset_min_id("covered_ngram_match_weight", var_idx)
    return var_idx


def generate_link_weights_nonnegative_constraints(a, I, J, b):
    global conn
    c = conn.cursor()
    c.execute("SELECT varid FROM link_similarity")
    for varid, in c:
        # A[constraint_idx, varid] >= 0
        a.append(-1.0)
        I.append(len(b))
        J.append(varid)
        b.append(0.0)


def generate_ngram_match_weight_sum_of_link_weights(v, P, Q, h):
    global conn
    c = conn.cursor()
    c2 = conn.cursor()

    # ref
    c.execute("SELECT varid, start, end FROM ngram_match_weight WHERE ref_or_para = 'r'")
    for varid, start, end in c:
        v.append(-1.0)
        P.append(len(h))
        Q.append(varid)

        c2.execute("SELECT varid FROM link_similarity WHERE ref_start = ? AND ref_end = ?", (start, end))
        for link_varid, in c2:
            v.append(1.0)
            P.append(len(h))
            Q.append(link_varid)

        h.append(0.0)

    # para
    c.execute("SELECT varid, start, end FROM ngram_match_weight WHERE ref_or_para = 'p'")
    for varid, start, end in c:
        v.append(-1.0)
        P.append(len(h))
        Q.append(varid)

        c2.execute("SELECT varid FROM link_similarity WHERE para_start = ? AND para_end = ?", (start, end))
        for link_varid, in c2:
            v.append(1.0)
            P.append(len(h))
            Q.append(link_varid)

        h.append(0.0)


def generate_ngram_match_weight_lt_one(a, I, J, b):
    global conn
    c = conn.cursor()
    c.execute("SELECT varid FROM ngram_match_weight")
    for varid, in c:
        a.append(1.0)
        I.append(len(b))
        J.append(varid)
        b.append(1.0)


def generate_covered_ngram_match_weight_lt_one(a, I, J, b):
    global conn
    c = conn.cursor()
    c.execute("SELECT varid FROM covered_ngram_match_weight")
    for varid, in c:
        a.append(1.0)
        I.append(len(b))
        J.append(varid)
        b.append(1.0)


def generate_covered_ngram_match_weight_lt_sum_ngram_match_weight(a, I, J, b):
    global conn
    c = conn.cursor()
    c2 = conn.cursor()

    c.execute("SELECT varid FROM covered_ngram_match_weight")
    for child_varid, in c:
        a.append(1.0)
        I.append(len(b))
        J.append(child_varid)

        c2.execute("""SELECT ngram_match_weight.varid
                      FROM covered_ngram_match_weight, ngram_match_weight
                      WHERE covered_ngram_match_weight.varid = ?
                        AND covered_ngram_match_weight.ref_or_para = ngram_match_weight.ref_or_para
                        AND covered_ngram_match_weight.start >= ngram_match_weight.start
                        AND covered_ngram_match_weight.end <= ngram_match_weight.end""",
                        (child_varid,))
        for parent_varid, in c2:
            a.append(-1.0)
            I.append(len(b))
            J.append(parent_varid)

        b.append(0.0)


def set_objective(nb_vars):
    # objective: to maximize 
    #
    # alpha * sum(covered_ngram_match_weight_ref) + beta *
    # sum(covered_ngram_match_weight_para)
    #
    # or,
    #
    # sum(covered_ngram_match_weight_ref) + fnum *
    # sum(covered_ngram_match_weight_para)
    #
    # we negate all terms so that the objective is minimized
    global fnum, conn
    c = [0.0] * nb_vars
    cursor = conn.cursor()
    cursor.execute("SELECT varid FROM covered_ngram_match_weight WHERE ref_or_para = 'r'")
    for varid, in cursor:
        c[varid] = -1.0
    cursor.execute("SELECT varid FROM covered_ngram_match_weight WHERE ref_or_para = 'p'")
    for varid, in cursor:
        c[varid] = -fnum
    return c


def process_line(ref_line, para_line):
    global conn
    set_up_sqlite()
    c = conn.cursor()

    # variables
    ref_char_pos_s = [token.rsplit('_', 1) for token in ref_line.split()]
    para_char_pos_s = [token.rsplit('_', 1) for token in para_line.split()]
    us = unit_link_similarity(ref_char_pos_s, para_char_pos_s)

    if len(us) == 0:
        # no match at all; skip
        return 0.0

    fill_in_link_similarity(ref_char_pos_s, para_char_pos_s, us)
    fill_in_ngram_match_weight_variables()
    fill_in_covered_ngram_match_weight()
    nb_vars = normalize_variable_ids()

    if nb_vars == 0:
        return 0.0

    # constraints: Ax <= b; Gx = h; minimize c.x
    #
    # A is a sparse matrix, represented by three vectors a, I, J, for its
    # values, rows and columns respectively. Similarly, G is a sparse matrix,
    # represented by three vectors v, P, Q respectively.
    #
    # c is an array.
    #
    # we have the following types of variables:
    #
    # - link_weight variables, e.g. how much weight is assigned to the link
    # between 雨伞多少 and 伞多.
    #
    # - ngram_match_weight variables, this is the sum of the relevant
    # link_weight variables, i.e. the total weight assigned to 雨伞多少. This
    # variable must be <= 1.
    # 
    # - covered_ngram_match_weight variables. This is the maximum of all
    # ngram_match_weight variables that covers this n-gram. e.g. the
    # unit_weight of 伞 <= sum(ngram_match_weight of 雨伞, 伞多, 雨伞多少, etc)
    #
    # there is a separate set of ngram_match_weight and ngram_match variables
    # for the reference and for the candidate.

    a = []
    I = []
    J = []
    b = []
    v = []
    P = []
    Q = []
    h = []
    c = []

    generate_link_weights_nonnegative_constraints(a, I, J, b)
    generate_ngram_match_weight_sum_of_link_weights(v, P, Q, h)
    generate_ngram_match_weight_lt_one(a, I, J, b)
    generate_covered_ngram_match_weight_lt_one(a, I, J, b)
    generate_covered_ngram_match_weight_lt_sum_ngram_match_weight(a, I, J, b)

    # the objective
    c = set_objective(nb_vars)

    global fnum
    max_objective = 0.0
    max_objective += 1.0 * len(list(iterate_ngrams(len(ref_char_pos_s))))
    max_objective += fnum * len(list(iterate_ngrams(len(para_char_pos_s))))

    # do the LP
    try:
        sol = solvers.lp(matrix(c), \
                spmatrix(a, I, J, (len(b), nb_vars)), matrix(b), \
                spmatrix(v, P, Q, (len(h), nb_vars)), matrix(h))

        if sol['status'] != 'optimal':
            print >> sys.stderr, sol
            # print >> sys.stderr, sol['x']
            # close enough; it's OK
            # raise Exception("Could not solve!")
    except:
        print >> sys.stderr, 'a', a
        print >> sys.stderr, 'I', I
        print >> sys.stderr, 'J', J
        print >> sys.stderr, 'b', b
        print >> sys.stderr, 'v', v
        print >> sys.stderr, 'P', P
        print >> sys.stderr, 'Q', Q
        print >> sys.stderr, 'h', h
        raise

    objective = -1.0 * sol['primal objective']
    return objective / max_objective


if __name__ == '__main__':
    parse_opts()
    for ref_line, para_line in zip(reference, paraphrase):
        score = process_line(ref_line, para_line)
        print >> outputfile, score
