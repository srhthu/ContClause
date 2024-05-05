"""
New version to calculate metrics.

Two type of metrics, collective and pointwise.

The expected input:
    ground truth: (pandas.DataFrame) fields of title, para_idx, answers
    prediction: (pandas.DataFrame) fields of title, para_idx, prediction

Note:
    the prediction should be already parsed. For example, "No" -> ""
    prediction has more rows than ground truth
"""
import pandas as pd
import numpy as np

from cont_gen.evaluate.token_metrics import get_point_counts

def safe_p_r_f1_iou_score(n_gold, n_pred, n_joint):
    """Handle the case of dividing zero"""
    if n_pred == 0:
        prec = 1.0
    else:
        prec = n_joint / n_pred
    
    if n_gold == 0:
        recall = 1.0
    else:
        recall = n_joint / n_gold
    
    if n_gold + n_pred == 0:
        f1 = 1
        iou = 1
    else:
        # f1 = 2 * (prec * recall) / (prec + recall)
        f1 = 2 * n_joint / (n_pred + n_gold)
        iou = n_joint / (n_gold + n_pred - n_joint)
    
    return prec, recall, f1, iou

def get_doc_metrics(whole_df: pd.DataFrame):
    """Get metrics for each document"""
    whole_gb = whole_df.groupby('title')

    def count2metric(df):
        # sum counts of all chunks
        count_s = df[['n_gold', 'n_pred', 'n_joint']].apply(np.sum)
        mets = safe_p_r_f1_iou_score(*count_s.to_list())
        return pd.Series(mets, index = ['p', 'r', 'f1', 'iou'])

    each_doc_metrics = whole_gb.apply(count2metric)
    doc_ave_metrics = each_doc_metrics.apply(np.mean)
    return doc_ave_metrics

def cal_collective_point_metrics(
        ground_df: pd.DataFrame, pred_df: pd.DataFrame,
        return_whole = False
        ):
    """
    collective metrics: first sum then cal metrics (docsum)
    pointwise metrics: average metrics of each document (docave)
    """
    ground_df = ground_df[['title', 'para_idx', 'answers']]
    pred_df = pred_df[['title', 'para_idx', 'prediction']]

    # concatenate all answer spans
    # assert isinstance(ground_df['answers'].iloc[0], list)
    # ground_df['answers'] = ground_df['answers'].apply(lambda k: ' '.join(k))

    # merge the two df on pred_df. For missing rows, fill answers to []
    joint_df = pred_df.merge(ground_df, how = 'left', on = ['title', 'para_idx'])

    joint_df.fillna('', inplace = True)

    count_df = joint_df.apply(
        lambda k: pd.Series(
            get_point_counts(k['answers'], k['prediction']),
            index = ['n_gold', 'n_pred', 'n_joint']),
        axis = 1
    )

    met_names = ['p', 'r', 'f1', 'iou']

    whole_df = joint_df.merge(count_df, left_index=True, right_index=True)
    doc_ave_metrics = get_doc_metrics(whole_df)
    doc_ave_metrics = dict(zip([f'docave_{k}' for k in met_names], doc_ave_metrics.to_list()))

    # get metrics

    # metric_df = count_df.apply(
    #     lambda k: safe_p_r_f1_iou_score(k['n_gold'], k['n_pred'], k['n_joint']),
    #     axis = 1, result_type='expand'
    # )
    
    # 
    # metric_df.columns = [f'macro_{k}' for k in met_names]

    # point_metrics = metric_df.apply(np.mean).to_dict()

    # Gather n_gold ... of all documents
    coll_count = count_df.apply(np.sum)
    coll_mets = safe_p_r_f1_iou_score(*coll_count.to_list())
    coll_metrics = dict(zip([f'docsum_{k}' for k in met_names], coll_mets))

    tot_metrics = {**doc_ave_metrics, **coll_metrics}
    if return_whole:
        return tot_metrics, whole_df
    else:
        return tot_metrics