import torch
from torch.utils import data
from torchsummary import summary
from dataset import get_dataset, HyperX
from MMPN import get_model
from train import test, train
from utils import get_device, sample_gt, compute_imf_weights, metrics, logger, display_dataset, display_goundtruth, dim_reduce
import argparse
import numpy as np
import warnings
import datetime
import visdom

warnings.filterwarnings("ignore")

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
parser.add_argument('--model', type=str, default='MMPN',
                    help="Model to train.")
parser.add_argument('--folder', type=str, default='../dataset/',
                    help="Folder where to store the "
                         "datasets (defaults to the current working directory).")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")
parser.add_argument('--run', type=int, default=1,
                    help="Running times.")

group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--sampling_mode', type=str, default='random',
                           help="Sampling mode (random sampling or disjoint, default:  fixed)")
group_dataset.add_argument('--training_percentage', type=float,
                           help="Percentage of samples to use for training")
group_dataset.add_argument('--validation_percentage', type=float,
                           help="In the training data set, percentage of the labeled data are randomly "
                                "assigned to validation groups.")
group_dataset.add_argument('--sample_nums', type=int, default=20,
                           help="Number of samples to use for training and validation")
group_dataset.add_argument('--load_data', type=str, default=None,
                           help="Samples use of training")

group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int,
                         help="Training epochs")
group_train.add_argument('--save_epoch', type=int, default=5,
                         help="Training save epoch")
group_train.add_argument('--patch_size', type=int,
                         help="patch size of  spectral feature extration model"
                              "(optional, if absent will be set by the model)")
group_train.add_argument('--patch_bands', type=int,
                         help="Segment the HSI data in the spectral dimension according to the size of patch_bands")
group_train.add_argument('--reserve_bands', type=int,
                         help="Spectral bands after PCA")
group_train.add_argument('--lr', type=float,
                         help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int,
                         help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride")

args = parser.parse_args()

RUN = args.run

# Dataset name
DATASET = args.dataset
# 生成日志
file_date = datetime.datetime.now().strftime('%Y-%m-%d')
log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
logger = logger('./logs/logs-' + file_date + DATASET + '.txt')
logger.info("---------------------------------------------------------------------")
logger.info("-----------------------------Next run log----------------------------")
logger.info("---------------------------{}--------------------------".format(log_date))
logger.info("---------------------------------------------------------------------")
# Device
CUDA_DEVICE = get_device(logger, args.cuda)
# Model name
MODEL = args.model
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Percentage of samples to use for training and validation
TRAINING_PERCENTAGE = args.training_percentage
VALIDATION_PERCENTAGE = args.validation_percentage
# Number of sample to use for training and validation
SAMPLE_NUMS = args.sample_nums
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
LOAD_DATA = args.load_data
TEST_STRIDE = args.test_stride

hyperparams = vars(args)

# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(logger, DATASET, FOLDER)
# 根据光谱划分的波段长度对原始数据进行填充
if hyperparams['reserve_bands'] is not None:
    img = dim_reduce(img, DATASET, reserve_components=hyperparams['reserve_bands'])
for i in range(RUN):
    if LOAD_DATA:
        train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
        test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
        train_gt = np.load(train_gt_file, 'r')
        logger.info("Load train_gt successfully!(PATH:{})".format(train_gt_file))
        logger.info("{} samples selected for training(over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
        logger.info("Training Percentage:{:.2}".format(np.count_nonzero(train_gt)/np.count_nonzero(gt)))
        test_gt = np.load(test_gt_file, 'r')
        logger.info("Load train_gt successfully!(PATH:{})".format(test_gt_file))
        logger.info("{} samples selected for training(over {})".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode=SAMPLING_MODE)
        logger.info("Save train_gt successfully!(PATH:{})".format(train_gt_file))
        logger.info("Save test_gt successfully!(PATH:{})".format(test_gt_file))
    logger.info("Running an experiment with the {} model, RUN [{}/{}]".format(MODEL, i + 1, RUN))
    logger.info("RUN:{}".format(i))
    # Open visdom server
    if SAMPLING_MODE == 'fixed':
        vis = visdom.Visdom(
            env='SAMPLENUMS' + str(SAMPLE_NUMS) + ' ' + DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(
                PATCH_SIZE) + ' ' + 'EPOCH' + str(EPOCH))
    else:
        vis = visdom.Visdom(
            env=DATASET + ' ' + MODEL + ' ' + 'PATCH_SIZE' + str(PATCH_SIZE) + ' ' + 'EPOCH' + str(EPOCH))
    if not vis.check_connection:
        print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

    # Number of classes
    N_CLASSES = len(LABEL_VALUES)
    # Number of spectral bands
    N_BANDS = img.shape[-1]
    # Instantiate the experiment based on predefined networks
    hyperparams.update(
        {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    
    # Class balancing
    if CLASS_BALANCING:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        # hyperparams.update({'weights': torch.from_numpy(weights)})
        weights = torch.from_numpy(weights.astype(np.float32))
        hyperparams['weights'] = weights.to(hyperparams['device'])

    # Get model
    model, optimizer, loss, hyperparams = get_model(DATASET, **hyperparams)

    # Sample random validation spectral
    val_gt, _ = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE,
                          sample_nums=SAMPLE_NUMS)

    logger.info("{} samples selected for validation(over {})".format(np.count_nonzero(val_gt), np.count_nonzero(gt)))

    logger.info("Running an experiment with the {} model".format(MODEL))


    # Generate the dataset
    train_dataset = HyperX(img, train_gt, **hyperparams)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   shuffle=True,
                                   drop_last=True)
    logger.info("Train dataloader:{}".format(len(train_loader)))

    val_dataset = HyperX(img, val_gt, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=hyperparams['batch_size'],
                                 drop_last=True)
    logger.info("Validation dataloader:{}".format(len(train_loader)))

    logger.info('----------Training parameters----------')

    for k, v in hyperparams.items():
        logger.info("{}:{}".format(k, v))
    logger.info("Network :")

    with torch.no_grad():
        for input, _ in train_loader:
            break
        summary(model.to(hyperparams['device']), input)
        
    total = sum([param.nelement() for param in model.parameters()])
    logger.info("Number of parameter: {}==>{:.2f}M".format(total, total / 1e6))
    
    if CHECKPOINT is not None:
        logger.info('Load model {} successfully!!!'.format(CHECKPOINT))
        model.load_state_dict(torch.load(CHECKPOINT))
    else:
        try:
            logger.info('----------Training process----------')
            train(logger=logger, net=model, optimizer=optimizer, criterion=loss, train_loader=train_loader,
                  epoch=hyperparams['epoch'], save_epoch=hyperparams['save_epoch'], scheduler=hyperparams['scheduler'],
                  device=hyperparams['device'], supervision=hyperparams['supervision'], val_loader=val_loader,
                  vis_display=vis, RUN=i)
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass
    prediction = np.asarray(test(model, img, hyperparams), np.uint8)
    display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(full)" + "RUN{}".format(i))
    results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0
    display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(semi)" + "RUN{}".format(i))

    logger.info('The network training successfully!!!')
    logger.info('----------Training result----------')
    logger.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
    logger.info("\nAccuracy:\n{}".format(results['Accuracy']))
    logger.info("\nF1 scores:\n{}".format(results['F1 scores']))
    logger.info("\nKappa:\n{}".format(results['Kappa']))
