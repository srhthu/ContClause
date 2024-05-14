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
from functools import partial

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

def sum_count_cal_metrics(df: pd.DataFrame, prefix = '')->pd.DataFrame:
    """
    Sum the counts over all df and get metrics.

    Args:
        df: fields of ['n_gold', 'n_pred', 'n_joint']
    """
    count_s = df[['n_gold', 'n_pred', 'n_joint']].apply(np.sum)
    mets = safe_p_r_f1_iou_score(*count_s.to_list())
    return pd.Series(mets, index = [prefix+k for k in ['p', 'r', 'f1', 'iou']])

def get_doc_metrics(whole_df: pd.DataFrame):
    """Get metrics for each document"""
    whole_gb = whole_df.groupby('title')

    each_doc_metrics = whole_gb.apply(partial(sum_count_cal_metrics, prefix = ''))
    doc_ave_metrics = each_doc_metrics.apply(np.mean)
    return doc_ave_metrics

def cal_collective_point_metrics(
        ground_df: pd.DataFrame, pred_df: pd.DataFrame,
        return_whole = False
        ):
    """
    Given the ground and pred dataframe, 
    collective metrics: first sum over all docs then cal metrics (docsum)
    pointwise metrics: average metrics of each document (docave)
    """
    # First, assert positive para are included in pred_df

    # Select columns to merge
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


def get_counts(ground_df, pred_df)->pd.DataFrame:
    """Count n_gold, n_pred and n_joint"""
    joint_df = pred_df.merge(ground_df, how = 'left', on = ['title', 'para_idx', 'q_id'])

    joint_df.fillna('', inplace = True)

    count_df = joint_df.apply(
        lambda k: pd.Series(
            get_point_counts(k['answers'], k['prediction']),
            index = ['n_gold', 'n_pred', 'n_joint']),
        axis = 1
    )
    whole_df = joint_df.merge(count_df, left_index=True, right_index=True)

    ## get the count information to save
    count_info = whole_df[['title', 'para_idx', 'q_id', 'n_gold', 'n_pred', 'n_joint']]
    return count_info

def get_overall_metrics(ground_df: pd.DataFrame, pred_df: pd.DataFrame, parse_fn, count_info = None):
    """
    Return all metrics include: 
        macro_{p, r, f1, iou}: for each clause type, first aggregate over all docs, 
                                then calculate metrics 
        micro_{p, r, f1, iou}: aggregate all docs and calculate metrics
        doc_macro{p, r, f1, iou}: for each clause type, calculate metrics for 
                                    each document, average over docs
    Args:
        ground_df: fileds of ['title', 'para_idx', 'q_id', 'answers', 'type']
        pred_df: fileds of ['title', 'para_idx', 'q_id', 'type', 'prediction']
        parse_fn: function to convert predictions
    """
    if count_info is None:
        # First, check the completeness of pred_df
        pos_ground = ground_df[ground_df['type'] > 0]
        gr_rec = set(map(tuple, pos_ground[['title', 'para_idx', 'q_id']].values.tolist()))
        pr_rec = set(map(tuple, pred_df[['title', 'para_idx', 'q_id']].values.tolist()))
        
        miss = [k for k in gr_rec if k not in pr_rec]
        if len(miss) > 0:
            raise ValueError(f'Error: {len(miss)} samples in ground_df'
                             f'({len(gr_rec)}) are not found in pred_df({len(pr_rec)})')
        
        # Parse prediction
        pred_df = pred_df.copy()
        pred_df['prediction'] = pred_df['prediction'].apply(parse_fn)

        count_info = get_counts(ground_df, pred_df)
    
    
    # Micro metrics
    micro_mets = sum_count_cal_metrics(count_info, prefix = 'micro_')

    # Macro metrics
    detail_q = count_info.groupby('q_id').apply(partial(sum_count_cal_metrics, prefix = 'macro_')).reset_index()
    macro_mets = detail_q.iloc[:,1:].apply(np.mean)

    # For two macro metrics
    group_by_q_doc = count_info.groupby(['q_id', 'title'])
    detail_q_doc_mets = group_by_q_doc.apply(partial(sum_count_cal_metrics, prefix = 'macro_')).reset_index()
    ## columns: q_id, title, macro_*
    
    detail_docq = detail_q_doc_mets.groupby('q_id').apply(lambda k: k.iloc[:, 1:].apply(np.mean))
    detail_docq = detail_docq.rename(lambda k: 'doc_' + k, axis = 'columns').reset_index()
    ## columns: q_id, doc_macro_*
    doc_macro_mets = detail_docq.iloc[:,1:].apply(np.mean)

    
    overall_mets = dict(**micro_mets.to_dict(), 
                        **macro_mets.to_dict(),
                        **doc_macro_mets.to_dict())
    detail_tab = detail_q.merge(detail_docq, how = 'outer', on = 'q_id')
    return overall_mets, detail_tab

