import pandas as pd
import numpy as np
import os
import cv2

def gen_exponential_sw_proposal(video_info, time_step=1, max_level=8, overlap=0.4,valid_num = 1):
    '''
    Generate exponential sliding windows
    '''
    spans = [2 ** x for x in range(max_level)]  ## 指数滑窗的滑窗长度
    duration = video_info.duration ## 视频时长
    pr = []
    for t_span in spans:
        span = t_span * time_step ## 拉伸
        step = int(np.ceil(span * (1 - overlap))) ## 向前移动的步长
        local_boxes = [(i, i + t_span) for i in np.arange(0, duration, step)] ## 对于当前滑窗长度span，i为 每个滑窗起始位置
        pr.extend(local_boxes) ## 滑窗起止点
    
    def valid_proposal(duration, span):
        '''
         Args: 
            duration: 完整时长
            span：滑窗起止点(start, end)
        '''
        real_span = min(duration, span[1]) - span[0]
        return real_span >= valid_num

    pr = list(filter(lambda x: valid_proposal(duration, x), pr)) ## filter(function:bool, iterable)，保留序列元素中返回True的元素
    return pr

def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])

def overlap_over_b(span_A, span_B):
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1]) ##交集
    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(span_B[1] - span_B[0])

def name_proposal(gt_spans, est_spans, thresh=0.0):
    """
    Assigng label to positive proposals
    :param gt_spans: [(label, (start, end)), ...] groundtruth
    :param est_spans: [(start, end), ...] sliding window
    :param thresh:
    :return: [(label, overlap, start, end), ...] same number of est_spans
    """
    ret = []
    for es in est_spans: ##对每个滑窗查找对应gt
        max_overlap = 0
        max_overlap_over_self = 0
        label = 0
        for gs in gt_spans:
            ov = temporal_iou(gs[1], es) ##iou
            ov_pr = overlap_over_b(gs[1], es)##交集占b的比
            if ov > thresh and ov > max_overlap: ##sliding window 的iou>阈值且是该gt对应滑窗覆盖最多的
                label = gs[0] 
                max_overlap = ov
                max_overlap_over_self = ov_pr
        ret.append((label, max_overlap, max_overlap_over_self, es[0], es[1]))

    return ret

def temporal_recall(gt_spans, est_spans, thresh=0.5):
    """
    Calculate temporal recall of boxes and estimated boxes
    Parameters
    ----------
    gt_spans: [(start, end), ...]
    est_spans: [(start, end), ...]

    Returns
    recall_info: (hit, total)
    -------

    """
    hit_slot = [False] * len(gt_spans)
    for i, gs in enumerate(gt_spans):
        for es in est_spans:
            if temporal_iou(gs, es) > thresh:
                hit_slot[i] = True
                break
    recall_info = (np.sum(hit_slot), len(hit_slot))
    return recall_info

def get_temporal_proposal_recall(pr_list, gt_list, thresh):
    recall_info_list = [temporal_recall(x, y, thresh=thresh) for x, y in zip(gt_list, pr_list)] ##（hit, total)
    per_video_recall = np.sum([x[0] == x[1] for x in recall_info_list]) / float(len(recall_info_list)) ##完整动作视频recall
    per_inst_recall = np.sum([x[0] for x in recall_info_list]) / float(np.sum([x[1] for x in recall_info_list]))##每个frame的recall
    return per_video_recall, per_inst_recall
