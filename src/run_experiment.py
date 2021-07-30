import bisect
from sklearn.multioutput import ClassifierChain
from numpy import uint
from utils.utils import set_logger
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import csv
import time
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
logger = set_logger("./log/experiment.log")


def train(X_train, y_train, model_type='ovr'):
    """ Do the training on the input transformed DataFrame and save model
    """
    start = time.time()
    logger.info("Training Beginning ...")
    if model_type == 'ovr':
        classifier = OneVsRestClassifier(

            LogisticRegression(penalty='l1', solver='liblinear', n_jobs=-1, class_weight={0: 1, 1: 4}, verbose=True))

    elif model_type == 'multioutput':
        forest = RandomForestClassifier(n_jobs=-1,
                                        class_weight={0: 1, 1: 3}, random_state=1)
        classifier = MultiOutputClassifier(forest, n_jobs=-1)
    else:  # Classifier Chains
        base_lr = LogisticRegression(
            penalty='l1', solver='liblinear', n_jobs=-1, class_weight={0: 1, 1: 3}, verbose=True)
        classifier = ClassifierChain(base_lr, order='random', random_state=42)

    classifier.fit(X_train, y_train)
    with open('model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    logger.info("Training over after {:.3f} seconds.".format(
        time.time() - start))


def test(X_test, threshold,  model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    start = time.time()
    logger.info("Testing Beginning ...")

    probas = classifier.predict_proba(X_test)
    y_pred = (probas[:, :] >= threshold).astype(int)
    # Now, when more than 5 labels are predicted, we keep only the ones having the 5 highest probabilities (usually 5 labels, unless some labels share the same probability)
    for i in range(y_pred.shape[0]):
        if sum(y_pred[i, :]) > 5:
            y_pred[i, :] = 0  # we set the whole row to 0
            # row_highest_probas will store the 5 highest probas in the row
            row_highest_probas = [0, 0, 0, 0, 0]
            for j in range(y_pred.shape[1]):
                if probas[i, j] > row_highest_probas[0]:  # if larger than the minimum
                    # save the position of the item in the row (column index)
                    #row_highest_probas_indices[bisect.bisect(row_highest_probas, probas[i,j])-1] = j
                    bisect.insort(row_highest_probas, probas[i, j])
                    row_highest_probas.pop(0)
            for j in range(y_pred.shape[1]):
                if probas[i, j] >= row_highest_probas[0]:
                    y_pred[i, j] = 1
        if sum(y_pred[i, :]) == 0:
            y_pred[i, np.argmax(probas[i, :])] = 1

    #y_pred = classifier.predict(X_test)
    with open('multi_label_binarizer.pkl', 'rb') as f:
        multi_label_binarizer = pickle.load(f)
    # print(len(multi_label_binarizer.classes_))
    #print(len(np.argwhere(np.all(y_pred[..., :] == 0, axis=0))))
    y_pred = multi_label_binarizer.inverse_transform(y_pred)
    y_pred_df = pd.DataFrame({"target": y_pred})
    y_pred_df['target'] = y_pred_df['target'].apply(list)
    y_pred_df.to_parquet('predictions.parquet')

    logger.info("Testing over after {:.1f} seconds.".format(
        time.time() - start))
    return y_pred_df
