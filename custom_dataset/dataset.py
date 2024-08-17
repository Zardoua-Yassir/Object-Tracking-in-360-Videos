# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os
from collections import namedtuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.bbox import center2corner, Center
from custom_dataset.augmentation import Augmentation
from config import cfg

general_logger = logging.getLogger('GeneralInfoLogger')
general_logger.setLevel(logging.INFO)

# File handler for general information logger
general_handler = logging.FileHandler('general_info.log')
general_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add the file handler to the general information logger
general_logger.addHandler(general_handler)

Corner = namedtuple('Corner', 'x1 y1 x2 y2')


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx, frames_subset=0):
        """
        Class to handle subdataset
        :param name: Name of the subdataset (e.g., LaSOT)
        :param root: path to the folder containing 511x511 crops. Used to produce various search and template images.
        :param anno: path to JSON annotation
        :param frame_range: the maximum number of consecutive frames separating the search and template frame.
        :param num_use: Number of videos to use from the current subdataset. If = -1, it is set to be the number of all
        videos in the subdatast.
        :param start_idx: an attribute used to figure out from which dataset a given index will be used to extract the
        sample
        :param frames_subset: if > 0, it's the number of frames to select from a given video. For instance, if we have
        a video named 'person-1' of 5200 frames, and frames_subset = 300, only the first 300 frames from 'person-1'
        will be used for training. Additional note: this argument is used to clip the list of values of key 'frames',
        the other keys containing all frames and their fine_bbox are not clipped since it's not necessary.
        """
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = root
        self.anno = os.path.join(cur_path, '../../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        self.frames_subset = frames_subset - 1  # - 1 to account for 0 index
        general_logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)  # JSON annotations as a dict. For example:
            # {'person-1': {'00': {'000000': [456, 346, 659, 631],
            #                       .....,
            #                      '000019': [513, 215, 712, 514]}}....,
            # {'person-2': {'00': {'000000': [573, 336, 620, 515],
            #                       .....
            #                     {'000019': [576, 334, 625, 516]}}}
            # NOTE: '00' here represents a track
            print("Json meta_data loaded")

        for video in list(meta_data.keys()):  # a list of video names (e.g., person-1, person-2, ...)
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int, filter(lambda x: x.isdigit(), frames.keys())))  # a list of frame indexes as INT
                # within the current track (e.g., "00"), which is itself within a video (e.g., "person-1")
                frames.sort()  # sort the list from smallest to biggest index value
                if self.frames_subset > 0:
                    frames = frames[0:self.frames_subset]
                meta_data[video][track]['frames'] = frames  # assign the sorted frame indices to the 'frames' key
                if len(frames) <= 0:  # log in a warning if there is no frame, then delete
                    general_logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:  # log in a warning if there is no video, then delete
                general_logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)  # Total videos in sub-dataset (currently = 2, accounts for person-1 and person-2)
        self.num_use = self.num if self.num_use == -1 else self.num_use  # if cfg.DATASET.DATASETNAME.NUM_USE = -1,
        # assign the number of videos to use from the subdataset (e.g., LaSOT) to the MAX (i.e., all videos). Else, use
        # cfg.DATASET.DATASETNAME.NUM_USE, which must be <= MAX
        self.videos = list(meta_data.keys())  # get the list of video names
        general_logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        general_logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        """
        Description of variables and commands:
        self.num: is the number of ALL videos in the current sub-dataset.
        self.num_use: is the number of videos to use from the current dataset, which can be <= self.num.
        lists: a list containing indexes of ALL VIDEOS in the current sub-dataset. From this list, self.num_use videos
        will be randomly picked, which is done by:
            1) shuffling the list 'lists'
            2) append the shuffled list to another list called 'pick'
            3) Once done, select 'self.num_use' indices (of videos) from 'pick' and return it.
        """
        lists = list(range(self.start_idx, self.start_idx + self.num))  # a list containing indexes of ALL VIDEOS in the
        # current sub-dataset
        pick = []  # this list will contain randomly shuffled indices of videos in the current sub-dataset
        while len(pick) < self.num_use:  # stop the selection of videos when the maximal number of videos to use is
            # reached
            np.random.shuffle(lists)  # random shuffling to allow random video selection
            pick += lists  # append shuffled indices to the list 'pick'
        return pick[:self.num_use]  # select self.num_use video indices and return them as a list

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        """
        Get a pair of template img and search img from the index-th video stored in self.videos
        :param index: video index
        :return:
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))  # randomly choose a track
        track_info = video[track]  # get GT b.boxes of all consecutive frames of the selected video

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))  # pick up a random frame from the selected track to extract
        # template
        left = max(template_frame - self.frame_range, 0)  # the max ensures not to select a negative frame index
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1  # the min ensures not to select an index
        # bigger than the last frame index of the selected track
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self, norm=True):
        """
        :param norm: normalize the images (search and template) into [0, 1] if True
        """
        print("Initializing the training dataset")
        self._norm = norm
        super(TrkDataset, self).__init__()

        # create sub UnitTestDummyDatasets
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                start
            )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        """
        Shuffling logic:
        1) Iterate over each sub_dataset stored in self.all_dataset
        2) For each sub_dataset, get its .pick list, which contains already shuffled indices of sub_dataset videos
        3) Append .pick to the list p
        4) Repeat for all subdatasets

        Shuffle the list 'p', then append it to 'pick' (not .pick).
        At this point, 'pick' contains shuffled video indices from all datasets, which are needed to complete one epoch.
        The while loop repeats the same process (by appending more indices to 'pick' from all datasets), until pick
        contains enough video indices to complete all epochs.

        Finally, this method returns a list whose length is equal to self.num (i.e., len(dataset)).
        :return:
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick  # .pick is an attribute of sub_dataset (e.g., LaSOT), whose length is equal to
                # the number of videos the user sets to pick from this subdataset)
                p += sub_p

            np.random.shuffle(p)
            pick += p
            m = len(pick)
        general_logger.info("shuffle done!")
        general_logger.info("Dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        """
        The authors define the length (number of all individual samples in all datasets) as:
        Number_of_videos_to_process_per_epoch * total_number_of_epochs (cfg.DATASET.VIDEOS_PER_EPOCH * cfg.TRAIN.EPOCH).
        In other words, the length of the dataset is tuned and controlled by the user through
        cfg.DATASET.VIDEOS_PER_EPOCH. Thus, an epoch does not necessarily iterate through the entire dataset (i.e., all
        videos and all frames).

        Also, since the frames of each video are randomly picked and distanced by the cfg.DATASET.DataName.FRAME_RANGE,
        processing cfg.DATASET.VIDEOS_PER_EPOCH videos per epoch does not mean processing all the frames of the selected
        videos.
        :return:
        """
        return self.num

    def __getitem__(self, index):
        """
        Get next item
        :param index: index of the next sample (search and template). This index is not reset to zero even if all batch
        samples are extracted. For instance, if we set the dataloader to a batch_size = 5 samples, index will keep
        getting incremented past 5 after all samples of a batch are extracted, and will never be reset to zero as long
        as the dataloader is being iterated over.

        It seems this index is overwritten by the index of
        the video within the dataset (e.g., index = 5 for person-5 within person sequences in LaSOT).
        :return:
        """
        # print(f"Getting {index}-th sample")
        # print("--")
        index = self.pick[index]  # self.pick is a list of shuffled video indexes to choose within a dataset.
        dataset, index = self._find_dataset(index)  # find the dataset object and index of video within this dataset. To
        # find out which dataset 'index' belongs to, the attribute .start_idx and .num of each sub_dataset is used.
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one UnitTestDummyDatasets
        if neg:  # # my tests indicate there will never be a negative sample.
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)
        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        # print("srch", search[0].split('\\')[-2:])
        # print("temp", template[0].split('\\')[-2:])
        # print("--")
        if template_image is None:
            print('error image:', template[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, gt_bbox = self.search_aug(search_image,
                                          search_box,
                                          cfg.TRAIN.SEARCH_SIZE,  # # = 255
                                          gray=gray)

        # cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)

        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)

        if self._norm:
            template = np.clip(template / 255.0, 0, 1)
            search = np.clip(search / 255.0, 0, 1)

        # p1 (top-left), p2 (bottom-right) of the object. Coordinates given within the search region search
        gt_bbox = np.array([gt_bbox.x1, gt_bbox.y1, gt_bbox.x2, gt_bbox.y2])

        labeled_data = {'template': template,
                        'search': search,
                        'gt_bbox': gt_bbox}

        return labeled_data
