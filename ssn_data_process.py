import pandas as pd
import numpy as np
import os
import cv2
import glob
from sw_utils import *

def parse_stage_config(stage_cfg):
    if isinstance(stage_cfg, int):
        return (stage_cfg,), stage_cfg
    elif isinstance(stage_cfg, tuple) or isinstance(stage_cfg, list):
        return stage_cfg, sum(stage_cfg)
    else:
        raise ValueError("Incorrect STPP config {}".format(stage_cfg))

class STPP():
    def __init__(self, annotation_path,mp_data_path,valid_num=6,overlap=0.8,IOU_thresh = [0.5, 0.7, 0.9],part_config=(1,(1,2),1),stride_pt=0.5):
        '''
        Args:
        annotation_path: it should include a csv format file with title [name,label,starting ,end], unit should be frame
        overlap：ratio of overlap between proposals(sliding windows)
        valid_num: length of proposals should > valid_num(frames)
        stride_pt: If stride_pt>1, then view it as fixed length of stride, or it will be viewed as ratio
        '''
        self.annt = pd.read_csv(annotation_path).dropna().iloc[:,1:]
        self.overlap = overlap
        self.valid_num=valid_num
        self.IOU_thresh = IOU_thresh
        self.mp_data_path = mp_data_path
        self.part_config = part_config
        self.stride_pt = stride_pt
    
    def create_gt(self):
        self.gt = self.annt.reset_index()
        self.action = self.gt['label'].unique()
        strides = self.action.shape[0]
        self.vid_num = self.gt.shape[0]
        self.gt['duration'] =0
        self.gt['label_num'] =0
        for i,a in enumerate(self.action):
            self.gt.iloc[self.gt[self.gt['label']==a].index,-1] = i+1
        a = self.gt.iloc[range(strides-1,self.vid_num,strides),:][['index','end']]
        cnt = 0
        for i in a['index']:
            self.gt.iloc[i-strides+1:i+1,5] = a.iloc[cnt,1]
            cnt+=1
        self.video_level = self.gt.iloc[range(strides-1,self.vid_num,strides),[1,5]]

        ## generate gt_spans
        self.gt_spans=[]
        for i in range(strides-1,self.gt.shape[0],strides):
            sub_seqence=[]
            for sub in np.abs(np.sort((-1)*np.array(range(strides)))):
                clip = self.gt.iloc[i-sub,:]
                num_label = clip['label_num']
                time_span = (clip['starting'],clip['end'])
                sub_seqence.append((num_label,time_span))
            self.gt_spans.append(sub_seqence)
        return 

    def gen_proposals(self):
        print('-'*5,'Generate proposals','-'*5)
        self.proposal_list = list(map(lambda x: gen_exponential_sw_proposal(self.video_level.iloc[x,:],valid_num=self.valid_num,overlap=self.overlap),range(self.video_level.shape[0])))
        print('average # of proposals: {} at overlap param {}'.format(np.mean(list(map(len, self.proposal_list))), 0.4))

        self.named_proposal_list = [name_proposal(x, y) for x,y in zip(self.gt_spans, self.proposal_list)]
        recall_list = []
        print("-"*15)
        for th in self.IOU_thresh:
            pv, pi = get_temporal_proposal_recall(self.proposal_list, [[y[1] for y in x] for x in self.gt_spans], th)
            print('IOU threshold {}. per video recall: {:02f}, per instance recall: {:02f}'.format(th, pv * 100, pi * 100))
            recall_list.append([0.4, th, np.mean(list(map(len, self.proposal_list))), pv, pi])

        print("average per video recall: {:.2f}, average per instance recall: {:.2f}".format(
            np.mean([x[3] for x in recall_list]), np.mean([x[4] for x in recall_list])))
        return 
    
    def seq_extraction(self,seq,videos,npy_path):
        vid_y = []
        vid_x=[]
        vid_start=[]
        vid_end=[]
        for i,vid in enumerate(videos):
            arr = np.load(os.path.join(npy_path,'{}.npy'.format(vid)))
            if len(seq[i])>0:
                item_y = []
                item_x = []
                item_start = []
                item_end = []
                for ind,span in enumerate(seq[i]):
                    duration = int(seq[i][ind][4])-int(seq[i][ind][3])
                    stride = int(duration*self.stride_pt) if self.stride_pt <=1 else int(self.stride_pt)
                    start_point = max(0,int(seq[i][ind][3])-stride)
                    end_point = min(arr.shape[0],int(seq[i][ind][4])+stride)
                    
                    item_x.append(arr[int(seq[i][ind][3]):int(seq[i][ind][4]),:])
                    #item_y.append([int(seq[i][ind][0])]*(int(seq[i][ind][4])-int(seq[i][ind][3])))
                    item_y.append(int(seq[i][ind][0]))
                    
                    item_start.append(arr[start_point:int(seq[i][ind][3]),:])
                    item_end.append(arr[int(seq[i][ind][4]):end_point,:])
                vid_y.append(item_y)
                vid_x.append(item_x)
                vid_start.append(item_start)
                vid_end.append(item_end)
        return vid_x, vid_y,vid_start,vid_end

    def get_stage_stpp(self, stage_ft, stage_parts, norm_num, scaling=None):
        '''
        计算池化特征
        Args:
            stage_ft: 当前stage（starting，course，end）的数据
            stage_parts：该stage分成几层，每层几个(starting_parts, course_parts, ending_parts)
            norm_num：该stage总unit数
        '''
        stage_stpp = []
        stage_len = stage_ft.shape[0] ##长度
        if stage_len >0:
            for n_part in stage_parts:
                ticks = np.arange(0, stage_len + 1, stage_len / n_part) ##n_part的起始点
                for i in range(n_part): 
                    part_ft = stage_ft[int(ticks[i]):int(ticks[i+1]), :].mean(axis=0)/norm_num
                    if scaling is not None:
                        part_ft = part_ft * scaling.resize(n_sample, 1)
                    stage_stpp.append(part_ft) ## (N sample, k part)
        else:
            for n_part in stage_parts:
                part_ft=np.array([0]*stage_ft.shape[1])
                stage_stpp.append(part_ft)
        return stage_stpp

    def classify_proposals(self,positive_iou=0.7,incom_iou=0.3,incom_over_self=0.8):
        self.positive_seq=[]
        self.bg_seq = []
        self.incomplete_seq = []

        for video in self.named_proposal_list:
            positive_seq_v=[]
            bg_seq_v = []
            incomplete_seq_v = []
            for i in video:
                if i[0]>0:
                    if i[1] > positive_iou:##iou>0.7
                        positive_seq_v.append(i)
                    elif (i[2]>incom_over_self) & (i[1]<incom_iou):
                        incomplete_seq_v.append(i)    
                else:
                    bg_seq_v.append(i)
            self.positive_seq.append(positive_seq_v)
            self.bg_seq.append(bg_seq_v)
            self.incomplete_seq.append(incomplete_seq_v)        
        return 
    
    def return_stpp_ft(self,li,x_seq):
        positive_x = x_seq[0]
        positive_start = x_seq[1]
        positive_end = x_seq[2]
        global_ft = []
        global_ft_course = []
        for i in range(li):
            for ind in range(len(positive_start[i])):
                feature_parts = [] ## 在列表中一次性添加另一个序列的多个值
                feature_parts.extend(self.get_stage_stpp(positive_start[i][ind], self.parts[0], self.norm_num[0], None))  # starting
                feature_parts.extend(self.get_stage_stpp(positive_x[i][ind], self.parts[1], self.norm_num[1], None))  # course
                feature_parts.extend(self.get_stage_stpp(positive_end[i][ind], self.parts[2], self.norm_num[2], None))  # ending
                stpp_ft = np.vstack(feature_parts)
                global_ft.append(stpp_ft)
                global_ft_course.append(np.vstack(self.get_stage_stpp(positive_x[i][ind], self.parts[1], self.norm_num[1], None)))
        return global_ft,global_ft_course

    def __call__(self,out_path,positive_iou=0.7,incom_iou=0.3,incom_over_self=0.8) :
        ## 1. Generate groundtruth related variables   
        self.create_gt()
        ## 2. Using sliding windows to generate proposal candidates
        self.gen_proposals()

        ## 3. Classify proposals as Positive, Incomplete, Background
        ## positive_seq, incomplete_seq, bg_seq
        self.classify_proposals(positive_iou,incom_iou,incom_over_self)

        ## 4. Split into parts
        starting_parts, starting_mult = parse_stage_config(self.part_config[0]) ## (1), 1
        course_parts, course_mult = parse_stage_config(self.part_config[1]) ##  (1,2),3
        ending_parts, ending_mult = parse_stage_config(self.part_config[2])
        feat_multiplier = starting_mult + course_mult + ending_mult ##各stage总unit数
        self.parts = (starting_parts, course_parts, ending_parts)
        self.norm_num = (starting_mult, course_mult, ending_mult)


        videos = self.gt['name'].unique()
        positive_x,positive_y,positive_start,positive_end = self.seq_extraction(self.positive_seq,videos,self.mp_data_path)
        incomplete_x,incomplete_y,incomplete_start,incomplete_end = self.seq_extraction(self.incomplete_seq,videos,self.mp_data_path)
        bg_x,bg_y,bg_start,bg_end  = self.seq_extraction(self.bg_seq,videos,self.mp_data_path)
        positive_seq_x = (positive_x,positive_start,positive_end)
        incomplete_seq_x  = (incomplete_x,incomplete_start,incomplete_end)
        bg_seq_x  = (bg_x,bg_start,bg_end)


        ## 5. Generate Global feature
        positive_ft,positive_course = self.return_stpp_ft(len(positive_start),positive_seq_x )
        bg_ft,bg_course = self.return_stpp_ft(len(bg_start),bg_seq_x )
        incomplete_ft,incomplete_course = self.return_stpp_ft(len(incomplete_start),incomplete_seq_x)

        global_ft = np.vstack([np.array(positive_ft),np.array(incomplete_ft)]) ##(785, 5, 132)
        global_y = np.hstack([np.hstack(positive_y),np.hstack(incomplete_y)]) ## (785,)
        global_ft_course = np.vstack([np.array(positive_course),np.array(incomplete_course),np.array(bg_course)]) ##(833, 3, 132)
        global_y_course = np.hstack([np.hstack(positive_y),np.hstack(incomplete_y),np.hstack(bg_y)])## (833,)

        
        out_dic={
            'global_ft':global_ft,
            'global_ft_course':global_ft_course,
            'global_y':global_y,
            'global_y_course':global_y_course
        }
        for name, value in out_dic.items():
            print('Shape of {}: '.format(name),value.shape)
            path = os.path.join(out_path,name)
            np.save(path,value)

        return 





if __name__ == '__main__':
    file_li = glob.glob(r'/home/sstc/文档/action_detection/test_2/out/cpp_combine/[0-9]*')
    ann_file_path = '/home/sstc/文档/action_detection/test_2/gym_annotation_frame.csv'
    mp_data_path = '/home/sstc/文档/action_detection/test_2/MP_Data/combine'
    #out_path = '/home/sstc/文档/action_detection/test_2/out/cpp_combine'
    for f in file_li:
        out_path = f
        stride_pt = float(out_path.split('/')[-1])
        a = STPP(ann_file_path,mp_data_path,valid_num=6,overlap=0.8,IOU_thresh = [0.5, 0.7, 0.9],part_config=(1,(1,2),1),stride_pt=stride_pt)
        a(out_path)
    print('-----------------done!----------------')