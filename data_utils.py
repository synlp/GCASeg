import os
import numpy as np
import torch

def readfile(filename):
    data = []
    sentence = []
    label = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                if len(sentence) > 0:
                    if len(sentence) < 500:
                        data.append((sentence, label))
                    sentence = []
                    label = []
                    verb_index = []
                continue
            splits = line.split()
            sentence.append(splits[0])
            tag = splits[1]
            label.append(tag)
        if 500 > len(sentence) > 0:
            data.append((sentence, label))
    return data

class DataProcessor():
    def __init__(self, tokenizer, labelmap, memory, gram2id=None, zen_ngram_dict=None, max_ngram_length=512):
        self.tokenizer = tokenizer
        self.memory = memory
        self.gram2id = gram2id
        self.labelmap = labelmap
        self.zen_ngram_dict = zen_ngram_dict
        self.max_ngram_length = max_ngram_length

    def get_features(self, data_dir, dataset):
        train_data_file = os.path.join(data_dir, dataset, 'train.tsv')
        train_examples = self.load_data(train_data_file)
        train_features = self.convert_examples_to_features(train_examples)

        dev_data_file = os.path.join(data_dir, dataset, 'dev.tsv')
        dev_examples = self.load_data(dev_data_file)
        dev_features = self.convert_examples_to_features(dev_examples)

        return train_features, dev_features

    def load_data(self, data_path, do_predict=False):
        if not do_predict:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        else:
            flag = 'predict'

        lines = readfile(data_path)

        examples = self.process_data(lines, flag)

        return examples

    def process_data(self, lines, flag):
        data = []
        for sentence, label in lines:
            if self.memory is not None:
                word_list = []
                matching_position = []
                for i in range(len(sentence)):
                    for j in range(self.max_ngram_length):
                        if i + j > len(sentence):
                            break
                        word = ''.join(sentence[i: i + j + 1])
                        if word in self.gram2id:
                            try:
                                index = word_list.index(word)
                            except ValueError:
                                word_list.append(word)
                                index = len(word_list) - 1
                            word_len = len(word)
                            for k in range(j + 1):
                                if word_len == 1:
                                    l = 'S'
                                elif k == 0:
                                    l = 'B'
                                elif k == j:
                                    l = 'E'
                                else:
                                    l = 'I'
                                matching_position.append((i + k, index, l))
            else:
                word_list = None
                matching_position = None
            data.append((sentence, label, word_list, matching_position))

        examples = []
        for i, (sentence, label, word_list, matching_position) in enumerate(data):
            guid = "%s-%s" % (flag, i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None,
                                         label=label, word=word_list, matrix=matching_position))
        return examples

    def convert_examples_to_features(self, examples):

        features = []

        length_list = []
        tokens_list = []
        labels_list = []
        valid_list = []
        label_mask_list = []

        word_num_list = []

        for (ex_index, example) in enumerate(examples):
            text_list = example.text_a
            label_list = example.label

            tokens = []
            labels = []
            valid = []
            label_mask = []

            for i, word in enumerate(text_list):
                token = self.tokenizer.tokenize(word)
                assert len(token) == 1
                tokens.extend(token)
                valid.append(1)
                labels.append(label_list[i])
                label_mask.append(1)

            assert len(tokens) == len(valid)

            length_list.append(len(tokens))
            tokens_list.append(tokens)
            labels_list.append(labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)

            if self.memory is not None:
                wordlist = example.word
                # wordlist = wordlist.split(' ') if len(wordlist) > 0 else []
                word_num_list.append(len(wordlist))

        seq_pad_length = max(length_list) + 2
        label_pad_length = seq_pad_length

        for indx, (example, tokens, labels, valid, label_mask) in \
                enumerate(zip(examples, tokens_list, labels_list, valid_list, label_mask_list)):

            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(self.labelmap["[CLS]"])

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(self.labelmap["[SEP]"])

            assert sum(valid) == len(label_ids)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_pad_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                label_ids.append(0)
                label_mask.append(0)

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(label_mask) == label_pad_length

            if self.memory is not None:
                wordlist = example.word
                # wordlist = wordlist.split(' ') if len(wordlist) > 0 else []
                matching_position = example.matrix

                max_word_size = max(word_num_list)
                word_ids = []
                matching_matrix = np.zeros((seq_pad_length, max_word_size), dtype=np.int)
                for word in wordlist:
                    try:
                        word_ids.append(self.gram2id[word])
                    except KeyError:
                        print(word)
                        print(wordlist)
                        raise KeyError()
                while len(word_ids) < max_word_size:
                    word_ids.append(0)
                for position in matching_position:
                    char_p = position[0] + 1
                    word_p = position[1]
                    if char_p > seq_pad_length - 2:
                        continue
                    else:
                        matching_matrix[char_p][word_p] = self.labelmap[position[2]]

                assert len(word_ids) == max_word_size
            else:
                word_ids = None
                matching_matrix = None

            if self.zen_ngram_dict is not None:
                ngram_matches = []
                #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
                max_gram_n = self.zen_ngram_dict.max_ngram_len

                for p in range(2, max_gram_n):
                    for q in range(0, len(tokens) - p + 1):
                        character_segment = tokens[q:q + p]
                        # j is the starting position of the ngram
                        # i is the length of the current ngram
                        character_segment = tuple(character_segment)
                        if character_segment in self.zen_ngram_dict.ngram_to_id_dict:
                            ngram_index = self.zen_ngram_dict.ngram_to_id_dict[character_segment]
                            ngram_matches.append([ngram_index, q, p, character_segment,
                                                  self.zen_ngram_dict.ngram_to_freq_dict[character_segment]])

                ngram_matches = sorted(ngram_matches, key=lambda s: s[-1], reverse=True)

                max_ngram_in_seq_proportion = math.ceil((len(tokens) / self.max_seq_length) * self.zen_ngram_dict.max_ngram_in_seq)
                if len(ngram_matches) > max_ngram_in_seq_proportion:
                    ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

                ngram_ids = [ngram[0] for ngram in ngram_matches]
                ngram_positions = [ngram[1] for ngram in ngram_matches]
                ngram_lengths = [ngram[2] for ngram in ngram_matches]
                ngram_tuples = [ngram[3] for ngram in ngram_matches]
                ngram_seg_ids = [0 if position < (len(tokens) + 2) else 1 for position in ngram_positions]

                ngram_mask_array = np.zeros(self.zen_ngram_dict.max_ngram_in_seq, dtype=np.bool)
                ngram_mask_array[:len(ngram_ids)] = 1

                # record the masked positions
                ngram_positions_matrix = np.zeros(shape=(seq_pad_length, self.zen_ngram_dict.max_ngram_in_seq), dtype=np.int32)
                for i in range(len(ngram_ids)):
                    ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

                # Zero-pad up to the max ngram in seq length.
                padding = [0] * (self.zen_ngram_dict.max_ngram_in_seq - len(ngram_ids))
                ngram_ids += padding
                ngram_lengths += padding
                ngram_seg_ids += padding
            else:
                ngram_ids = None
                ngram_positions_matrix = None
                ngram_lengths = None
                ngram_tuples = None
                ngram_seg_ids = None
                ngram_mask_array = None

            feature = {"input_ids":input_ids,
                       "input_mask":input_mask,
                       "segment_ids":segment_ids,
                       "label_id":label_ids,
                       "valid_ids":valid,
                       "label_mask":label_mask,
                       "word_ids":word_ids,
                       "matching_matrix":matching_matrix,
                       "ngram_ids":ngram_ids,
                       "ngram_positions":ngram_positions_matrix,
                       "ngram_lengths":ngram_lengths,
                       "ngram_tuples":ngram_tuples,
                       "ngram_seg_ids":ngram_seg_ids,
                       "ngram_masks":ngram_mask_array,
                       }
            for k,v in feature.items():
                if v is None:
                    continue
                feature[k] = torch.tensor(v, dtype=torch.long)

            features.append(feature)
        return features

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 word=None, matrix=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

        self.word = word
        self.matrix = matrix




