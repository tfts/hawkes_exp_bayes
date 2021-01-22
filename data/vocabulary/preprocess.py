'''
0. Download the dataset underlying the paper 'B. Settles and B. Meeder, "A trainable spaced repetition model for language learning," in ACL, 2016.' from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/N8XJME. For more on the dataset format, refer to https://github.com/duolingo/halflife-regression
1. Extract the gz file.
2. Extract only users learning english words (i.e., produce the file "learning_traces_en.csv") with "awk -F , '$5=="en"' settles.acl16.learning_traces.13m.csv > learning_traces_en.csv".
3. This script extracts only users learning the words of ranking in RANKINGS (while excluding multi-ranking words)
'''
import numpy as np
import pandas as pd

RANKINGS_DICT = {'b1b2': {'B1', 'B2'}, 'a1a2c1c2': {'A1', 'A2', 'C1', 'C2'}}

cerf_en = pd.read_csv('cerf_en/cerf_en.csv') # NOTE this assumes the existence of a list of words and their corresponding rankings - see script in that folder that produces such a list
for RANKINGS in ['b1b2', 'a1a2c1c2']:
    result = ''
    with open('learning_traces_en.csv', 'r') as f:
        for line_i, line in enumerate(f):
            lexemes_raw = line.strip().split(',')[7]
            lexemes = lexemes_raw[:lexemes_raw.index('<')].split('/')
            for lexeme in lexemes:
                if lexeme in cerf_en['word'].values:
                    cerf_en_lexeme = cerf_en[cerf_en['word'] == lexeme]
                    lexeme_ranking = cerf_en_lexeme['level'].iloc[0]
                    # NOTE with the first part of the if I exclude multi-ranking words. To include them, consider: "cerf_en[cerf_en['word'] == 'and'].sort_values('level')"
                    if cerf_en_lexeme.shape[0] == 1 and lexeme_ranking in RANKINGS_DICT[RANKINGS]:
                        result += lexeme_ranking + ',' + lexeme + ',' + line
                        break
            if line_i % 50000 == 0:
                print(line_i)
                with open('learning_traces_en_{}.csv'.format(RANKINGS), 'a') as g:
                    g.write(result)
                result = ''
