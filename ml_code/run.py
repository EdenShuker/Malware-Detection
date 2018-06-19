import os
import sys
from os.path import isdir
import yaml

try:
    conf = yaml.load(open('ml_code/config_ml.yaml', 'r'))
except:
    print('Error')
    sys.exit()

""" FEATURES SELECTION """

ngrams_dir = 'ml_code/ngrams'
if not isdir(ngrams_dir):
    os.mkdir(ngrams_dir, 0755)

for i in range(conf['num_classes']):
    print(i)
    sys.argv = ['ml_code/extract_ngrams.py', '-n', i, '-p', 'data/train_set.csv']
    execfile('ml_code/extract_ngrams.py')

features_dir = 'ml_code/features'
if not isdir(features_dir):
    os.mkdir(features_dir, 0755)

sys.argv = ['ml_code/join_ngrams.py', conf['num_classes']]
execfile('ml_code/join_ngrams.py')

""" CREATE FEATURE VECTORS """

f2v_dir = 'ml_code/f2v'
if not isdir(f2v_dir):
    os.mkdir(f2v_dir, 0755)

sys.argv = ['ml_code/f2v.py', 'data/train_set.csv', 'ml_code/f2v/train.f2v', conf['dir_benign_dll']]
execfile('ml_code/f2v.py')

sys.argv = ['ml_code/f2v.py', 'data/test_set.csv', 'ml_code/f2v/test.f2v', conf['dir_benign_dll']]
execfile('ml_code/f2v.py')

""" RUN MODEL """

sys.argv = ['ml_code/model.py']
if conf['train']:
    sys.argv += ['-train', 'data/train_set.csv', 'ml_code/f2v/train.f2v']
    if (conf['save']):
        sys.argv += ['-save', 'ml_code/' + conf['model_save_name']]
elif conf['load']:
    sys.argv += ['-load', 'ml_code/' + conf['model_load_name']]
if conf['test']:
    sys.argv += ['-test', 'data/test_set.csv', 'ml_code/f2v/test.f2v', 'ml_code/test.output']

execfile('ml_code/model.py')

""" EVALUATE MODEL """

# on train set
sys.argv = ['ml_code/eval_model.py', 'data/train_set.csv', 'ml_code/train.output']
if conf['show_matrix']:
    sys.argv += ['-show-matrix']
execfile('ml_code/eval_model.py')

# on train set
sys.argv = ['ml_code/eval_model.py', 'data/test_set.csv', 'ml_code/test.output']
if conf['show_matrix']:
    sys.argv += ['-show-matrix']
execfile('ml_code/eval_model.py')
