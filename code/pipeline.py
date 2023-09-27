import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import Levenshtein
from annotation import Annotation
from post import AnnotatedPost

def preprocess_text_lite(text):
    text = str(text).lower()
    sentences = sent_tokenize(text)
    return sentences


def is_negated(sentence_piece, found_symptom_expression):
    neg_trigs = open('../lexicon/neg_trigs.txt').read()
    neg_trigs = [phrase.strip() for phrase in neg_trigs.split('\n')]
    negation_prefix_regex_pattern = '(' + '|'.join(neg_trigs) + r')(\b\W*\b\w*\b\W*\b){0,2}\w*'
    negation_regex_pattern = negation_prefix_regex_pattern + found_symptom_expression
    final_expression = found_symptom_expression
    negated = False
    match = re.search(negation_regex_pattern, sentence_piece)
    negation_start = None
    if match is not None:
        negated = True
        final_expression = match.group()
        negation_start = match.start()
    return negated, final_expression, negation_start


def find_exact_matches(lexicon, symptom_vocab, sentences, with_negation=True):
    output_vector = [0] * len(symptom_vocab)
    sentence_pieces_exact_removed = []
    phrases_found = []
    cui_list = []
    expressions = [key.lower() for key in lexicon.keys()]
    regex_pattern = r"\b(" + "|".join(expressions) + r")\b"
    symptom_regex = re.compile(regex_pattern)
    for sent in sentences:
        prev_match_end = 0
        for match in symptom_regex.finditer(sent):
            if with_negation:
                negated, phrase_with_negation, negation_start = is_negated(
                    sent[prev_match_end:match.end()], match.group())
                if negated:
                    this_match_start = negation_start + prev_match_end
                    this_phrase_found = phrase_with_negation
                    output_vector[symptom_vocab.index(lexicon[match.group()] + "-1")] += 1
                    cui_list.append(lexicon[match.group()] + "-1")
                else:
                    this_match_start = match.start()
                    this_phrase_found = match.group()
                    output_vector[symptom_vocab.index(lexicon[match.group()] + "-0")] += 1
                    cui_list.append(lexicon[match.group()] + "-0")
            else:
                output_vector[symptom_vocab.index(lexicon[match.group()])] += 1
                cui_list.append(lexicon[match.group()])
                this_match_start = match.start()
                this_phrase_found = match.group()
            sentence_pieces_exact_removed.append(sent[prev_match_end:this_match_start])
            prev_match_end = match.end()
            phrases_found.append(this_phrase_found)
        if prev_match_end < len(sent)-1:
            sentence_pieces_exact_removed.append(sent[prev_match_end:])
    return sentence_pieces_exact_removed, phrases_found, output_vector, cui_list


def __find_fuzzy_matches_helper(tokenized_sentence_pieces, expression, window_size, threshold, with_negation=True):
    phrases_found = []
    tokenized_sentence_pieces_matches_removed = []
    negations = []
    for tokenized_sent in tokenized_sentence_pieces:
        prev_match_end_ind = 0
        window_start_ind = 0
        while window_start_ind + window_size <= len(tokenized_sent):
            window_end_ind = window_start_ind + window_size
            this_window_phrase = " ".join(tokenized_sent[window_start_ind:window_end_ind])
            if Levenshtein.ratio(expression.lower(), this_window_phrase) > threshold:
                if with_negation:
                    this_sentence_frag = " ".join(tokenized_sent[prev_match_end_ind:window_end_ind])
                    negated, final_expression, negation_start = is_negated(this_sentence_frag, this_window_phrase)
                    if negated:
                        phrases_found.append(final_expression)
                        negations.append(1)
                        this_match_start = negation_start + prev_match_end_ind
                    else:
                        phrases_found.append(this_window_phrase)
                        negations.append(0)
                        this_match_start = window_start_ind
                else:
                    phrases_found.append(" ".join(tokenized_sent[window_start_ind:window_end_ind]))
                    this_match_start = window_start_ind
                if this_match_start > prev_match_end_ind:
                    tokenized_sentence_pieces_matches_removed.append(
                        tokenized_sent[prev_match_end_ind:this_match_start])
                prev_match_end_ind = window_end_ind
                window_start_ind += window_size
            else:
                window_start_ind += 1
        if prev_match_end_ind < len(tokenized_sent)-1:
            tokenized_sentence_pieces_matches_removed.append(tokenized_sent[prev_match_end_ind:])
    return phrases_found, tokenized_sentence_pieces_matches_removed, negations


def find_fuzzy_matches(lexicon, symptom_vocab, sentences, T_i, k, with_negation=True):
    output_vector = [0] * len(symptom_vocab)
    sentence_pieces_fuzzy_removed = []
    phrases_found = []
    tokenized_sentence_pieces = []
    cui_list = []
    for sent in sentences:
        tokenized_sent = word_tokenize(sent)
        tokenized_sentence_pieces = [tokenized_sent]
        for expression in lexicon.keys():
            try:
                tokenized_expression = word_tokenize(expression)
            except:
                tokenized_expression = expression
                print(f"Failed to tokenize {expression}")
            window_min = len(tokenized_expression) - 1  # number of words
            window_max = len(tokenized_expression) + 2
            dynamic_thresh = T_i - k * len(tokenized_expression) / 100
            for window_size in range(window_min, window_max + 1):
                these_phrases_found, tokenized_sentence_pieces, negations = __find_fuzzy_matches_helper(
                    tokenized_sentence_pieces, expression, window_size, dynamic_thresh)
                if len(these_phrases_found) > 0:
                    phrases_found += these_phrases_found
                for i, phrase in enumerate(these_phrases_found):
                    if with_negation:
                        if negations[i] == 0:
                            output_vector[symptom_vocab.index(lexicon[expression] + "-0")] += 1
                            cui_list.append(lexicon[expression] + "-0")
                        else:
                            output_vector[symptom_vocab.index(lexicon[expression] + "-1")] += 1
                            cui_list.append(lexicon[expression] + "-1")
                    else:
                        output_vector[symptom_vocab.index(lexicon[expression])] += 1
                        cui_list.append(lexicon[expression])
    return tokenized_sentence_pieces, phrases_found, output_vector, cui_list


def get_multilabel_metrics(truth_dict, pred_dict):
    tp = 0
    fp = 0
    fn = 0
    for post_id in truth_dict.keys():
        for ind in range(len(truth_dict[post_id])):
            if truth_dict[post_id][ind] == pred_dict[post_id][ind]:
                tp += truth_dict[post_id][ind]
            elif truth_dict[post_id][ind] > pred_dict[post_id][ind]:
                fn += truth_dict[post_id][ind] - pred_dict[post_id][ind]
                tp += pred_dict[post_id][ind]
            else:
                fp += pred_dict[post_id][ind] - truth_dict[post_id][ind]
                tp += pred_dict[post_id][ind]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * recall * precision) / (recall + precision)
    return recall, precision, f1


def grid_search_for_fuzzy_params(truth_vectors, posts, lexicon, symptom_vocab):
    T_i = np.arange(0.4, 1.1, 0.1)
    k = np.arange(0, 5, 1)
    T_i_grid, k_grid = np.meshgrid(T_i, k)
    f1s = np.empty_like(T_i_grid)
    for i in range(T_i_grid.shape[0]):
        print(f"Searching k = {k_grid[i][0]}")
        for j in range(T_i_grid.shape[1]):
            predicted_vectors = dict()
            for post in posts:
                sentences = preprocess_text_lite(post.text)
                sentence_pieces_exact_removed, exact_phrases_found, exact_output_vector, cui_list_exact = find_exact_matches(
                    lexicon, symptom_vocab, sentences, with_negation=False)
                tokenized_sentence_pieces, fuzzy_phrases_found, fuzzy_output_vector, cui_list_fuzzy = find_fuzzy_matches(
                    lexicon, symptom_vocab, sentence_pieces_exact_removed, T_i=T_i_grid[i, j], k=k_grid[i, j],
                    with_negation=False)
                predicted_vectors[post.get_id()] = np.add(exact_output_vector, fuzzy_output_vector)
            recall, precision, f1s[i, j] = get_multilabel_metrics(truth_vectors, predicted_vectors)
    best_ind = np.argmax(f1s)
    best_ind_tuple = np.unravel_index(best_ind, f1s.shape)
    best_T_i = T_i_grid[best_ind_tuple]
    best_k = k_grid[best_ind_tuple]
    best_f1 = f1s[best_ind_tuple]
    return best_T_i, best_k, best_f1


def run_pipeline(posts, lexicon, symptom_vocab, standard_symptoms):
    annotated_posts = []
    for i, post in enumerate(posts):
        print(f"Processing post {i}/{len(posts)}")
        text = post.text
        # text processing
        sentences = preprocess_text_lite(text)
        sentence_pieces_exact_removed, exact_phrases_found, exact_output_vector, cui_list_exact = find_exact_matches(
            lexicon, symptom_vocab, sentences, with_negation=True)
        sentence_pieces_fuzzy_removed, fuzzy_phrases_found, fuzzy_output_vector, cui_list_fuzzy = find_fuzzy_matches(
            lexicon, symptom_vocab, sentence_pieces_exact_removed, T_i=0.87, k=2, with_negation=True)
        # output formatting
        cuis_no_flag = [cui[:-2] for cui in cui_list_exact+cui_list_fuzzy]
        these_standard_symptoms = [standard_symptoms[cui] for cui in cuis_no_flag]
        negations = [cui[-1] for cui in cui_list_exact+cui_list_fuzzy]
        this_annotation = Annotation(post_id=post.get_id(), expressions=exact_phrases_found+fuzzy_phrases_found,
                                     standard_symptoms=these_standard_symptoms, cuis=cuis_no_flag, negations=negations)
        this_annotated_post = AnnotatedPost(post, this_annotation)
        annotated_posts.append(this_annotated_post)
    print("Finished symptom detection")
    return annotated_posts


def save_annotated_posts_to_xlsx(annotated_posts, save_path):
    print("Saving results...")
    annotated_post_dict_list = [ann_post.to_dict() for ann_post in annotated_posts]
    annotated_posts_df = pd.DataFrame.from_dict(annotated_post_dict_list)
    annotated_posts_df.to_excel(save_path)