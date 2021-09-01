from __future__ import unicode_literals
import numpy as np
from collections import Counter
from latent_dialog.utils import Pack
import json
from nltk.tokenize import WordPunctTokenizer
import logging

PAD = '<pad>'
UNK = '<unk>'
USR = 'YOU:'
SYS = 'THEM:'
BOD = '<d>'
EOD = '</d>'
BOS = '<s>'
EOS = '<eos>'
SEL = '<selection>'
SPECIAL_TOKENS_DEAL = [PAD, UNK, USR, SYS, BOD, EOS]
SPECIAL_TOKENS = [PAD, UNK, USR, SYS, BOS, BOD, EOS, EOD]
STOP_TOKENS = [EOS, SEL]
DECODING_MASKED_TOKENS = [PAD, UNK, USR, SYS, BOD]


class DealCorpus(object):
    def __init__(self, config):
        self.config = config
        self.train_corpus = self._read_file(self.config.train_path)
        self.val_corpus = self._read_file(self.config.val_path)
        self.test_corpus = self._read_file(self.config.test_path)
        self._extract_vocab()
        self._extract_goal_vocab()
        self._extract_outcome_vocab()
        print('Loading corpus finished.')

    def _read_file(self, path):
        with open(path, 'r') as f:
            data = f.readlines()

        return self._process_dialogue(data)

    def _process_dialogue(self, data):
        def transform(token_list):
            usr, sys = [], []
            ptr = 0
            while ptr < len(token_list):
                turn_ptr = ptr
                turn_list = []
                while True:
                    cur_token = token_list[turn_ptr]
                    turn_list.append(cur_token)
                    turn_ptr += 1
                    if cur_token == EOS:
                        ptr = turn_ptr
                        break
                all_sent_lens.append(len(turn_list))
                if turn_list[0] == USR:
                    usr.append(Pack(utt=turn_list, speaker=USR))
                elif turn_list[0] == SYS:
                    sys.append(Pack(utt=turn_list, speaker=SYS))
                else:
                    raise ValueError('Invalid speaker')

            all_dlg_lens.append(len(usr) + len(sys))
            return usr, sys

        new_dlg = []
        all_sent_lens = []
        all_dlg_lens = []
        for raw_dlg in data:
            raw_words = raw_dlg.split()

            # process dialogue text
            cur_dlg = []
            words = raw_words[raw_words.index('<dialogue>') + 1: raw_words.index('</dialogue>')]
            words += [EOS]
            usr_first = True
            if words[0] == SYS:
                words = [USR, BOD, EOS] + words
                usr_first = True
            elif words[0] == USR:
                words = [SYS, BOD, EOS] + words
                usr_first = False
            else:
                print('FATAL ERROR!!! ({})'.format(words))
                exit(-1)
            usr_utts, sys_utts = transform(words)
            for usr_turn, sys_turn in zip(usr_utts, sys_utts):
                if usr_first:
                    cur_dlg.append(usr_turn)
                    cur_dlg.append(sys_turn)
                else:
                    cur_dlg.append(sys_turn)
                    cur_dlg.append(usr_turn)
            if len(usr_utts) - len(sys_utts) == 1:
                cur_dlg.append(usr_utts[-1])
            elif len(sys_utts) - len(usr_utts) == 1:
                cur_dlg.append(sys_utts[-1])

            # process goal (6 digits)
            # FIXME FATAL ERROR HERE !!!
            cur_goal = raw_words[raw_words.index('<partner_input>') + 1: raw_words.index('</partner_input>')]
            # cur_goal = raw_words[raw_words.index('<input>')+1: raw_words.index('</input>')]
            if len(cur_goal) != 6:
                print('FATAL ERROR!!! ({})'.format(cur_goal))
                exit(-1)

            # process outcome (6 tokens)
            cur_out = raw_words[raw_words.index('<output>') + 1: raw_words.index('</output>')]
            if len(cur_out) != 6:
                print('FATAL ERROR!!! ({})'.format(cur_out))
                exit(-1)


            # process movie
            user_movie = raw_words[raw_words.index('<user>') + 1: raw_words.index('</user>')]

            new_dlg.append(Pack(dlg=cur_dlg, goal=cur_goal, out=cur_out, movie = user_movie))

        print('Max utt len = %d, mean utt len = %.2f' % (
            np.max(all_sent_lens), float(np.mean(all_sent_lens))))
        print('Max dlg len = %d, mean dlg len = %.2f' % (
            np.max(all_dlg_lens), float(np.mean(all_dlg_lens))))
        return new_dlg

    def _extract_vocab(self):
        all_words = []
        for dlg in self.train_corpus:
            for turn in dlg.dlg:
                all_words.extend(turn.utt)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c in vocab_count])

        print('vocab size of train set = %d,\n' % (raw_vocab_size,) + \
              'cut off at word %s with frequency = %d,\n' % (vocab_count[-1][0], vocab_count[-1][1]) + \
              'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_words),))
        self.vocab = SPECIAL_TOKENS_DEAL + [t for t, cnt in vocab_count if t not in SPECIAL_TOKENS_DEAL]
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.vocab_dict[UNK]

        global DECODING_MASKED_TOKENS
        from string import ascii_letters, digits
        letter_set = set(list(ascii_letters + digits))
        vocab_list = [t for t, cnt in vocab_count]
        masked_words = []
        for word in vocab_list:
            tmp_set = set(list(word))
            if len(letter_set & tmp_set) == 0:
                masked_words.append(word)
        # DECODING_MASKED_TOKENS += masked_words
        print('Take care of {} special words (masked).'.format(len(masked_words)))

    def _extract_goal_vocab(self):
        all_goal = []
        for dlg in self.train_corpus:
            all_goal.extend(dlg.goal)
        vocab_count = Counter(all_goal).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c in vocab_count])

        print('goal vocab size of train set = %d, \n' % (raw_vocab_size,) + \
              'cut off at word %s with frequency = %d, \n' % (vocab_count[-1][0], vocab_count[-1][1]) + \
              'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_goal),))
        self.goal_vocab = [UNK] + [g for g, cnt in vocab_count]
        self.goal_vocab_dict = {t: idx for idx, t in enumerate(self.goal_vocab)}
        self.goal_unk_id = self.goal_vocab_dict[UNK]

    def _extract_outcome_vocab(self):
        all_outcome = []
        for dlg in self.train_corpus:
            all_outcome.extend(dlg.out)
        vocab_count = Counter(all_outcome).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c in vocab_count])

        print('outcome vocab size of train set = %d, \n' % (raw_vocab_size,) + \
              'cut off at word %s with frequency = %d, \n' % (vocab_count[-1][0], vocab_count[-1][1]) + \
              'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_outcome),))
        self.outcome_vocab = [UNK] + [o for o, cnt in vocab_count]
        self.outcome_vocab_dict = {t: idx for idx, t in enumerate(self.outcome_vocab)}
        self.outcome_unk_id = self.outcome_vocab_dict[UNK]

    def get_corpus(self):
        id_train = self._to_id_corpus('Train', self.train_corpus)
        id_val = self._to_id_corpus('Valid', self.val_corpus)
        id_test = self._to_id_corpus('Test', self.test_corpus)
        return id_train, id_val, id_test

    def _to_id_corpus(self, name, data):
        results = []
        for dlg in data:
            if len(dlg.dlg) < 1:
                continue
            id_dlg = []
            for turn in dlg.dlg:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker)
                id_dlg.append(id_turn)
            id_goal = self._goal2id(dlg.goal)
            id_out = self._outcome2id(dlg.out)
            results.append(Pack(dlg=id_dlg, goal=id_goal, out=id_out, movie = dlg.movie))
        return results

    def _sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]

    def _goal2id(self, goal):
        return [self.goal_vocab_dict.get(g, self.goal_unk_id) for g in goal]

    def _outcome2id(self, outcome):
        return [self.outcome_vocab_dict.get(o, self.outcome_unk_id) for o in outcome]

    def sent2id(self, sent):
        return self._sent2id(sent)

    def goal2id(self, goal):
        return self._goal2id(goal)

    def outcome2id(self, outcome):
        return self._outcome2id(outcome)

    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def id2goal(self, id_list):
        return [self.goal_vocab[i] for i in id_list]

    def id2outcome(self, id_list):
        return [self.outcome_vocab[i] for i in id_list]

