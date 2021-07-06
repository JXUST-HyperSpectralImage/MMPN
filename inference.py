import argparse
from RepMLP import get_model
from dataset import get_dataset
from train import test
from utils import get_device, sample_gt, compute_imf_weights, metrics, logger, display_dataset, display_goundtruth
import numpy as np
import visdom
import torch
import datetime


parser = argparse.ArgumentParser(description="Run experiments on various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='IndianPines',
                    help="Choice one dataset for training"
                         "Dataset to train. Available:\n"
                         "PaviaU"
                         "HoustonU"
                         "IdianPines"
                         "KSC"
                         "Botswana"
                         "Salinas")
parser.add_argument('--model', type=str, default='RepMLP',
                    help="Model to train.")
parser.add_argument('--folder', type=str, default='../dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")
parser.add_argument('--patch_size', type=int,
                         help="patch size of  spectral feature extration model"
                              "(optional, if absent will be set by the model)")
parser.add_argument('--batch_size', type=int,
                         help="Batch size (optional, if absent will be set by the model")
parser.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--test_gt', action='store_true',
                           help="Samples use of testing")

args = parser.parse_args()
MODEL = args.model
TEST_GT = args.test_gt
CHECKPOINT = args.restore
DATASET = args.dataset
PATCH_SIZE = args.patch_size
TEST_STRIDE = args.test_stride
FOLDER = args.folder
# 生成日志
file_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
logger = logger('./logs/logs-' + file_date + DATASET + '.txt')
logger.info("---------------------------------------------------------------------")
logger.info("-----------------------------Next run log----------------------------")
logger.info("---------------------------{}--------------------------".format(log_date))
logger.info("---------------------------------------------------------------------")
CUDA_DEVICE = get_device(logger, args.cuda)
hyperparams = vars(args)
img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(logger, DATASET, FOLDER)
test_gt_file = '../dataset/' + DATASET + '/test_gt' + str(0) + '.npy'
test_gt = np.load(test_gt_file, 'r')
logger.info("Load train_gt successfully!(PATH:{})".format(test_gt_file))
logger.info("{} samples selected for training(over {})".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
vis = visdom.Visdom(env=DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(PATCH_SIZE))
N_CLASSES = len(LABEL_VALUES)
N_BANDS = img.shape[-1]
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

model, _, _, hyperparams = get_model(DATASET, **hyperparams)
model.load_state_dict(torch.load(CHECKPOINT))

prediction = test(model, img, hyperparams)
display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(full)" + "RUN{}".format(i))
results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
mask = np.zeros(gt.shape, dtype='bool')
for l in IGNORED_LABELS:
    mask[gt == l] = True
prediction[mask] = 0
display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(semi)" + "RUN{}".format(i))
