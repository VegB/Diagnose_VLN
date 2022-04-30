import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import csv
import base64
import time
from utils.logger import write_to_record_file


class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size, record_file):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

        print("Start loading the image feature")
        start = time.time()
        views = 36
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
        features = {}
        with open(self.img_ft_file, "r") as tsv_in_file:     # Open the tsv file.
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
            for item in reader:
                long_id = item['scanId'] + "_" + item['viewpointId']
                features[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                                    dtype=np.float32).reshape((views, -1))   # Feature of long_id is (36, 512)
        self._feature_store = features
        write_to_record_file("Finish Loading the image feature from %s in %0.4f seconds" % (img_ft_file, time.time() - start), record_file)

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        ft = self._feature_store[key]
        return ft


def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if dataset == 'r2r':
                with open(os.path.join(anno_dir, 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r2r_last':
                with open(os.path.join(anno_dir, 'LastSent', 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r2r_back':
                with open(os.path.join(anno_dir, 'ReturnBack', 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r4r':
                with open(os.path.join(anno_dir, 'R4R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'rxr':
                new_data = []
                with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_guide_enc_xlmr.jsonl'%split)) as f:
                    for item in f:
                        new_data.append(item)
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)

        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512, args=None):
    data = []
    for i, item in enumerate(load_datasets(splits, args)): 
        new_item = dict(item)
        new_item['instr_id'] = item['instruction_id']
        new_item['instruction'] = item['instructions'][0]
        instr = new_item['instruction']
        instr_tokens = ['[CLS]'] + tokenizer.tokenize(instr)[:max_instr_len-2] + ['[SEP]']
        new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(instr_tokens)
        data.append(new_item)
    return data


def get_label_for_setting(setting):
    """
    Label is the abbrev of the setting.
    'replace_object' -> 'ro'
    """
    return ''.join(w[0] for w in setting.split('_'))


def load_datasets(splits, args):
    """

    :param splits: A list of split.
        if the split is "something@5000", it will use a random 5000 data from the data
    :return:
    """
    import random
    data = []
    old_state = random.getstate()

    dataset = args.dataset
    setting = args.setting
    rate = args.rate
    repeat_idx = args.repeat_idx

    record_file = args.log_filepath 
    
    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        if setting in ['default', 'mask_env', 'numeric_default']:
            instr_setting = setting if setting != 'mask_env' else 'default'
            filename = os.path.join(args.data_dir, instr_setting, f'{args.dataset}_{split}.json')
        else:
            label = get_label_for_setting(setting)
            filename = os.path.join(args.data_dir, setting, f'{label}{rate:.2f}_{repeat_idx}', f'{dataset}_{split}.json')

        with open(filename) as f:
            new_data = json.load(f)
        write_to_record_file('Load data from %s' % filename, record_file)

        # Partition
        if number > 0:
            random.seed(0)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

 
def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    return sim

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0, minus_elevation=False):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    if minus_elevation:
        base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    else:
        base_elevation = 0
        
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size, minus_elevation=False):
    return [get_point_angle_feature(
        sim, angle_feat_size, baseViewId, minus_elevation=minus_elevation
        ) for baseViewId in range(36)]

