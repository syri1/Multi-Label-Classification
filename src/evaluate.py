from utils.utils import set_logger
from sklearn.metrics import hamming_loss, jaccard_score, precision_score, recall_score, classification_report
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logger = set_logger("./log/evaluation.log")


def evaluate(y_true, y_pred, show_report=False):
    # y_true and y_pred are one column DataFrames, we have to fit a MultLabelBinarizer on y_test U y_pred and transform them before we can evaluate
    logger.info("Starting evaluation")
    y_true.columns = ['target']
    y_all = pd.concat([y_true, y_pred], ignore_index=True)

    y_all = MultiLabelBinarizer().fit_transform(y_all['target'])

    y_true_binarized, y_pred_binarized = y_all[:len(
        y_true)], y_all[len(y_true):]
    assert len(y_true) == len(y_true_binarized) and len(
        y_pred) == len(y_pred_binarized)
    logger.info("The results on the test set are : ")
    logger.info("Hamming Loss : {:.3f} -- Jaccard Similarity Score : {:.3f} ".format(
        hamming_loss(y_true_binarized, y_pred_binarized), jaccard_score(y_true_binarized, y_pred_binarized, average='micro')))
    logger.info("Precision Score : {:.3f} -- Recall Score : {:.3f} ".format(precision_score(
        y_true_binarized, y_pred_binarized, average='micro'), recall_score(y_true_binarized, y_pred_binarized, average='micro')))
    if show_report:
        logger.info(classification_report(y_true_binarized, y_pred_binarized))
    logger.info("Evaluation over")
