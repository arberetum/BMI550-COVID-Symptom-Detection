import os

import post
from annotation import Annotation
import symptom
from post import Post, AnnotatedPost
from pipeline import run_pipeline, grid_search_for_fuzzy_params, save_annotated_posts_to_xlsx
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold



def combine_all_annotations(save_path = None):
    """ Combine annotations done by students in the class, randomly choosing a set of annotations for posts that were
    annotated more than once

    :param save_path: where to the save the resulting Excel file
    :return:
    """
    # add all the annotations
    annotated_post_dictionary = defaultdict(list)
    ann_file_list = os.listdir("../data/annots")
    for ann_file in ann_file_list:
        annot_df = pd.read_excel(os.path.join("../data/annots", ann_file))
        for index, row in annot_df.iterrows():
            if row["Symptom Expressions"] is not None:
                expressions = row["Symptom Expressions"].split("$$$")[1:-1]
                standard_symptoms = row["Standard Symptom"].split("$$$")[1:-1]
                cuis = row["Symptom CUIs"].split("$$$")[1:-1]
                negations = row["Negation Flag"].split("$$$")[1:-1]
                curr_post = Post(row["ID"], row["DATE"], row["TEXT"])
                try:
                    curr_annotation = Annotation(row["ID"], expressions, standard_symptoms, cuis, negations)
                except:
                    print(f"Issue in row {index} of {ann_file}, skipping...")
                annotated_post_dictionary[curr_post].append(curr_annotation)
    output_dict = defaultdict(list)
    rand_state = np.random.RandomState(1234)
    for post in annotated_post_dictionary.keys():
        output_dict["ID"].append(post.get_id())
        output_dict["DATE"].append(post.date)
        output_dict["TEXT"].append(post.text)
        random_annot = rand_state.choice(annotated_post_dictionary[post])
        annot_dict = random_annot.to_dict()
        output_dict["Symptom Expressions"].append(annot_dict["Symptom Expressions"])
        output_dict["Standard Symptom"].append(annot_dict["Standard Symptom"])
        output_dict["Symptom CUIs"].append(annot_dict["Symptom CUIs"])
        output_dict["Negation Flag"].append(annot_dict["Negation Flag"])
        output_dict["Annotation Count"].append(len(annotated_post_dictionary[post]))
    output_df = pd.DataFrame.from_dict(output_dict)
    if save_path is not None:
        output_df.to_excel(save_path, index=False)
    return output_df


def run_grid_search(train_df, symptom_vocab):
    train_vectors_dict_no_neg = dict()
    for index, row in train_df.iterrows():
        symptom_cuis = row["Symptom CUIs"].split("$$$")[1:-1]
        this_vec = [0] * len(symptom_vocab)
        for cui in symptom_cuis:
            if cui in symptom_vocab:
                this_vec[symptom_vocab.index(cui)] += 1
        train_vectors_dict_no_neg[row["ID"]] = this_vec
    post_ids = train_df["ID"].to_list()
    kf3 = KFold(n_splits=3, random_state=42, shuffle=True)
    for i, (train_inds, test_inds) in enumerate(kf3.split(post_ids)):
        fold_train_vectors = dict()
        fold_train_posts = []
        for ind in train_inds:
            post_id = post_ids[ind]
            fold_train_vectors[post_id] = train_vectors_dict_no_neg[post_id]
            fold_train_posts.append(Post(post_ids[ind],
                                         train_df.loc[train_df["ID"] == post_id, "DATE"].iloc[0],
                                         train_df.loc[train_df["ID"] == post_id, "TEXT"].iloc[0]))
        best_T_i, best_k, best_f1 = grid_search_for_fuzzy_params(fold_train_vectors, fold_train_posts, covid_lexicon,
                                                                 symptom_vocab)
        print(f"Best T_i for fold {i}: {best_T_i}")
        print(f"Best k for fold {i}: {best_k}")
        print(f"Best f1-score for fold {i}: {best_f1}")


if __name__ == '__main__':
    # if os.path.exists("../data/StudentAnnots.xlsx"):
    #     train_df = pd.read_excel("../data/StudentAnnots.xlsx")
    # else:
    #     train_df = combine_all_annotations("../data/StudentAnnots.xlsx")

    symptom_vocab = open("../lexicon/cuilist.txt").read().split()
    symptom_vocab.append("C0000000")
    symptom_vocab_with_neg = []
    for symptom in symptom_vocab:
        symptom_vocab_with_neg.append(symptom + "-0")
        symptom_vocab_with_neg.append(symptom + "-1")

    covid_lexicon_lines = open('../lexicon/COVID-Twitter-Symptom-Lexicon.txt').readlines()
    covid_lexicon = dict()
    standard_symptoms = dict()
    for line in covid_lexicon_lines:
        items = line.strip().split('\t')
        covid_lexicon[items[2].lower()] = items[1]
        standard_symptoms[items[1]] = items[0]

    # run_grid_search(train_df, symptom_vocab)

    test_df = pd.read_excel("../data/Assignment1GoldStandardSet.xlsx")
    test_posts = []
    for i, row in test_df.iterrows():
        this_post = Post(id=row["ID"], date="", text=row["TEXT"])
        test_posts.append(this_post)

    annotated_posts = run_pipeline(test_posts, covid_lexicon, symptom_vocab_with_neg, standard_symptoms)
    save_annotated_posts_to_xlsx(annotated_posts, "../results/Assignment1GoldStandardSet_Results.xlsx")
