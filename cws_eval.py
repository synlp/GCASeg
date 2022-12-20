
def eval_sentence(y_pred, y, sentence):
    # words = sentence.split(' ')
    seg_pred = []
    word_pred = ''

    if y is not None:
        word_true = ''
        seg_true = []
        for i in range(len(y)):
            word_true += sentence[i]
            if y[i] in ['S', 'E']:
                seg_true.append(word_true)
                word_true = ''
        seg_true_str = ' '.join(seg_true)
    else:
        seg_true_str = None

    for i in range(len(y_pred)):
        word_pred += sentence[i]
        if y_pred[i] in ['S', 'E']:
            seg_pred.append(word_pred)
            word_pred = ''
    seg_pred_str = ' '.join(seg_pred)
    return seg_true_str, seg_pred_str


def cws_evaluate_word_PRF(y_pred, y, label_list=None):
    #dict = {'E': 2, 'S': 3, 'B':0, 'I':1}
    y_pred_new = []
    y_new = []
    for y_p, y_t in zip(y_pred, y):
        assert len(y_p) == len(y_t)
        y_pred_new.extend(y_p)
        y_new.extend(y_t)
    cor_num = 0
    if label_list is not None:
        y_pred_new = [label_list[label_id-1] for label_id in y_pred_new]
        y_new = [label_list[label_id-1] for label_id in y_new]
    yp_wordnum = y_pred_new.count('E') + y_pred_new.count('S')
    yt_wordnum = y_new.count('E') + y_new.count('S')
    start = 0
    for i in range(len(y_new)):
        if y_new[i] == 'E' or y_new[i] == 'S':
            flag = True
            for j in range(start, i+1):
                if y_new[j] != y_pred_new[j]:
                    flag = False
            if flag:
                cor_num += 1
            start = i+1

    P = cor_num * 100 / float(yp_wordnum) if yp_wordnum > 0 else -1
    R = cor_num * 100 / float(yt_wordnum) if yt_wordnum > 0 else -1

    F = 2 * P * R / (P + R) if (P + R) > 0 else -1
    return P, R, F


def cws_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id):
    cor_num = 0
    yt_wordnum = 0
    for y_pred, y, sentence in zip(y_pred_list, y_list, sentence_list):
        start = 0
        for i in range(len(y)):
            if y[i] == 'E' or y[i] == 'S':
                word = ''.join(sentence[start:i+1])
                if word in word2id:
                    start = i + 1
                    continue
                flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y[j] != y_pred[j]:
                        flag = False
                if flag:
                    cor_num += 1
                start = i + 1

    OOV = cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    return OOV
