''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter
import numpy as np
import networkx as nx
import subprocess
import itertools
import base64
import heapq
import random
import cv2
from nltk.corpus import wordnet as wn
import torch

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>', '<BOS>']

vocab_pad_idx = base_vocab.index('<PAD>')
vocab_unk_idx = base_vocab.index('<UNK>')
vocab_eos_idx = base_vocab.index('<EOS>')
vocab_bos_idx = base_vocab.index('<BOS>')

def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    file_path = os.path.dirname(__file__)
    # connect_folder = os.path.abspath(os.path.join(file_path,'..','..','connectivity'))
    connect_folder = os.path.abspath(os.path.join(file_path,'..','R2R-EnvDrop', 'connectivity'))
    for scan in scans:
        with open('%s/%s_connectivity.json' % (connect_folder,scan)) as f:
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


def get_label_for_setting(setting):
    """
    Label is the abbrev of the setting.
    'replace_object' -> 'ro'
    """
    return ''.join(w[0] for w in setting.split('_'))


def load_datasets(args, splits):
    data = []

    dataset = args.dataset
    setting = args.setting
    rate = args.rate
    repeat_idx = args.repeat_idx

    for split in splits:
        # It only needs some part of the dataset?
        components = split.split("@")
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json
        if setting in ['default', 'mask_env', 'dynamic_mask_env', 'numeric_default']:
            instr_setting = setting if setting not in ['mask_env', 'dynamic_mask_env'] else 'default'
            filename = os.path.join(args.data_dir, instr_setting, f'{args.dataset}_{split}.json')
        else:
            label = get_label_for_setting(setting)
            filename = os.path.join(args.data_dir, setting, f'{label}{rate:.2f}_{repeat_idx}', f'{dataset}_{split}.json')

        with open(filename) as f:
            new_data = json.load(f)
        print('Load data from %s' % filename)
        
        # Partition
        if number > 0:
            random.seed(0)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number]

        # Join
        data += new_data

    return data

def decode_base64(string):
    if sys.version_info[0] == 2:
        return base64.decodestring(bytearray(string))
    elif sys.version_info[0] == 3:
        return base64.decodebytes(bytearray(string, 'utf-8'))
    else:
        raise ValueError("decode_base64 can't handle python version {}".format(sys.version_info[0]))

class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab=None):
        self.vocab = vocab
        self.word_to_index = {}
        self.index_is_verb = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
                self.index_is_verb[i] = int(Tokenizer.is_verb(word))

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def filter_verb(self, toks, sel_verb=False):
        is_verb = [self.index_is_verb[tok] for tok in toks]
        if sel_verb:
            sel_indexes = [i for i,x in enumerate(is_verb) if x]
        else:
            sel_indexes = [i for i,x in enumerate(is_verb) if not x]
        return is_verb, sel_indexes

    @staticmethod
    def is_verb(word):
        if word in base_vocab:
            return True
        for _entry in wn.synsets(word):
            if _entry.name().split('.')[0] == word and _entry.pos() == 'v':
                return True
        return False

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []
        for word in Tokenizer.split_sentence(sentence):
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(vocab_unk_idx)
        #encoding.append(vocab_eos_idx)
        #utterance_length = len(encoding)
        #if utterance_length < self.encoding_length:
            #encoding += [vocab_pad_idx] * (self.encoding_length - len(encoding))
        #encoding = encoding[:self.encoding_length] # leave room for unks
        arr = np.array(encoding)
        return arr, len(encoding)

    def decode_sentence(self, encoding, break_on_eos=False, join=True):
        sentence = []
        for ix in encoding:
            if ix == (vocab_eos_idx  if break_on_eos else vocab_pad_idx):
                break
            else:
                sentence.append(self.vocab[ix])
        if join:
            return " ".join(sentence)
        return sentence


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab):
    ''' Build a vocab, starting with base vocab containing a few useful tokens. '''
    count = Counter()
    data = load_datasets(splits)
    for item in data:
        for instr in item['instructions']:
            count.update(Tokenizer.split_sentence(instr))
    vocab = list(start_vocab)
    for word,num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab),path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def k_best_indices(arr, k, sorted=False):
    # https://stackoverflow.com/a/23734295
    if k >= len(arr):
        if sorted:
            return np.argsort(arr)
        else:
            return np.arange(0, len(arr))
    ind = np.argpartition(arr, -k)[-k:]
    if sorted:
        ind = ind[np.argsort(arr[ind])]
    return ind

def structured_map(function, *args, **kwargs):
    #assert all(len(a) == len(args[0]) for a in args[1:])
    nested = kwargs.get('nested', False)
    acc = []
    for t in zip(*args):
        if nested:
            mapped = [function(*inner_t) for inner_t in zip(*t)]
        else:
            mapped = function(*t)
        acc.append(mapped)
    return acc


def flatten(lol):
    return [l for lst in lol for l in lst]

def all_equal(lst):
    return all(x == lst[0] for x in lst[1:])

def try_cuda(pytorch_obj):
    import torch.cuda
    try:
        disabled = torch.cuda.disabled
    except:
        disabled = False
    if torch.cuda.is_available() and not disabled:
        return pytorch_obj.cuda()
    else:
        return pytorch_obj

def pretty_json_dump(obj, fp):
    json.dump(obj, fp, sort_keys=True, indent=4, separators=(',', ':'))

def spatial_feature_from_bbox(bboxes, im_h, im_w):
    # from Ronghang Hu
    # https://github.com/ronghanghu/cmn/blob/ff7d519b808f4b7619b17f92eceb17e53c11d338/models/spatial_feat.py

    # Generate 5-dimensional spatial features from the image
    # [xmin, ymin, xmax, ymax, S] where S is the area of the box
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))
    # Check the size of the bounding boxes
    assert np.all(bboxes[:, 0:2] >= 0)
    assert np.all(bboxes[:, 0] <= bboxes[:, 2])
    assert np.all(bboxes[:, 1] <= bboxes[:, 3])
    assert np.all(bboxes[:, 2] <= im_w)
    assert np.all(bboxes[:, 3] <= im_h)

    feats = np.zeros((bboxes.shape[0], 5), dtype=np.float32)
    feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
    feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
    feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
    feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
    feats[:, 4] = (feats[:, 2] - feats[:, 0]) * (feats[:, 3] - feats[:, 1]) # S
    return feats

def run(arg_parser, entry_function):
    arg_parser.add_argument("--pdb", action='store_true')
    arg_parser.add_argument("--ipdb", action='store_true')
    arg_parser.add_argument("--no_cuda", action='store_true')
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--save_args", action='store_false')

    args = arg_parser.parse_args()
    print('parameters:')
    print(json.dumps(vars(args), indent=2))

    args = DotDict(vars(args))
    args.RESULT_DIR = os.path.join(args.snap_dir, args.experiment_name,'results')
    args.SNAPSHOT_DIR = os.path.join(args.snap_dir, args.experiment_name,'snapshots')
    args.PLOT_DIR = os.path.join(args.snap_dir, args.experiment_name,'plots')
    make_dirs([args.RESULT_DIR, args.SNAPSHOT_DIR, args.PLOT_DIR])

    import torch.cuda
    torch.cuda.disabled = args.no_cuda

    def log(out_file):
        ''' git updates '''
        subprocess.call("git rev-parse HEAD", shell=True, stdout=out_file)
        subprocess.call("git --no-pager diff", shell=True, stdout=out_file)
        out_file.write('\n\n')
        out_file.write(' '.join(sys.argv))
        out_file.write('\n\n')
        json.dump(dict(args), out_file)
        out_file.write('\n\n')

    #log(sys.stdout)
    if args.save_args:
        import datetime
        now = datetime.datetime.now()
        args_fn = 'args-%d-%d-%d,%d:%d:%d' % (now.year,now.month,now.day,now.hour,now.minute,now.second)
        with open(os.path.join(args.PLOT_DIR, args_fn), 'w') as f:
            log(f)

    # Make the folders
    args.RESULT_DIR = os.path.join(args.snap_dir, args.experiment_name,'results')
    args.SNAPSHOT_DIR = os.path.join(args.snap_dir, args.experiment_name,'snapshots')
    args.PLOT_DIR = os.path.join(args.snap_dir, args.experiment_name,'plots')
    make_dirs([args.RESULT_DIR, args.SNAPSHOT_DIR, args.PLOT_DIR])

    def _call_entry_function(entry_function, args):
        if args.ipdb:
            import ipdb
            ipdb.runcall(entry_function, args)
        elif args.pdb:
            import pdb
            pdb.runcall(entry_function, args)
        else:
            entry_function(args)

    if args.setting == 'default' and not args.reset_img_feat:
        _call_entry_function(entry_function, args)
    else:
        for repeat_idx in range(args.repeat_time):
            args.repeat_idx = repeat_idx
            _call_entry_function(entry_function, args)


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color='green', bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def filter_param(m):
    return [p for p in m.parameters() if p.requires_grad]

def module_grad(module, requires_grad=False):
    for p in module.parameters():
        p.requires_grad_(requires_grad)

def make_dirs(list_of_dirs):
    for directory in list_of_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

class PriorityQueue:
    def __init__(self, max_size=0, maxHeap=True):
        self.queue = []
        self.priority = []
        self.pri = []
        self.maxHeap = maxHeap
        self.locked = False
        self.len = 0
        assert(max_size==0)

    def lock(self):
        self.locked = True

    def push(self, item, priority):
        if self.locked:
            return
        self.queue.append(item)
        self.priority.append(priority)

        p = priority.item() if type(priority) is torch.Tensor else priority
        if self.maxHeap:
            p = - p
        heapq.heappush(self.pri, (p, self.len))
        self.len += 1

    def pop(self):
        if self.locked:
            return 0, self.priority[0], self.queue[0]
        if len(self.pri) == 0:
            print("PriorityQueue error: pop from an empty queue")
            import pdb;pdb.set_trace()
        p, idx = heapq.heappop(self.pri)
        item = self.queue[idx]
        priority = self.priority[idx]
        return idx, priority, item

    def peak(self):
        if len(self.pri) == 0:
            return None
        p, idx = self.pri[0]
        return idx, self.priority[idx], self.queue[idx]

    def size(self):
        return len(self.pri)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def check_dir(dir):
    if os.path.exists(dir):
        print(f'{dir} exists!')
    else:
        os.mkdir(dir)
        print(f'{dir} created!')


def _match_object(object, list_of_objects):
    """Check if obj is contained in a list of objects, or is match to some object in the list"""
    if object in ['floor', 'wall', 'ceiling']:
        return False
    for obj in set(list_of_objects):
        if obj in object:
            return True
    return False


def load_bbox(scanId, viewpointId, instruction_objects, args):
    """
    Load bbox json.
    Mask out portions of objects.
    Reorganize bbox.
    """
    mask_rate = args.rate
    bbox_file = args.bbox_pattern % (scanId, viewpointId)
    data = json.load(open(bbox_file, 'r'))

    mentioned_objects = []
    all_appeared_obj_ids = []
    for idx in range(36):
        current_objects = data[str(idx)]
        for obj_id, item in current_objects.items():
            all_appeared_obj_ids.append(obj_id)
            if _match_object(item['name'], instruction_objects):
                mentioned_objects.append(obj_id)
    mentioned_objects = list(set(mentioned_objects))
    mask_num = int(mask_rate*len(mentioned_objects))
    masked_objects = random.sample(mentioned_objects, mask_num)

    if args.img_feat_mode == 'dynamic_controlled_trial':
        all_appeared_obj_ids = list(set(all_appeared_obj_ids))
        masked_objects = random.sample(all_appeared_obj_ids, mask_num)

    rst = [[] for _ in range(36)]
    for idx in range(36):
        current_objects = data[str(idx)]
        for obj_id, item in current_objects.items():
            if obj_id in masked_objects:
                rst[idx].append(item)

    return rst


def mask_objects(img, to_be_masked_objs_info, scanId, viewpointId, ix):
    img = np.array(img, copy=True)

    for obj_info in to_be_masked_objs_info:
        average = img.mean(axis=0).mean(axis=0)  # fill mask with average color
        s_x, s_y, w, h = obj_info['bbox2d']
        e_x = s_x + w
        e_y = s_y + h
        thickness = -1
        img = cv2.rectangle(img, (s_x, s_y), (e_x, e_y), average, thickness)

    return img


def transform_img(im):
    ''' Prep opencv 3 channel image for the network '''
    im = np.array(im, copy=True)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[103.1, 115.9, 123.2]]]) # BGR pixel mean
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, :, :, :] = im_orig
    blob = blob.transpose((0, 3, 1, 2))
    return blob 
