"""
This script trains the SiamCAR model
"""
import time

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn.utils import clip_grad_norm_
from models.models import SiamCarV1
from config import cfg
from custom_dataset.dataset import TrkDataset
from toolbox.loss_functions import SiamCarLoss
import os
from utils.lr_scheduler import build_lr_scheduler
import re
import math
from toolbox.misc import TrainTimer

import logging

# logger = logging.getLogger('Global Info')

# Configure the first logger for general information
general_logger = logging.getLogger('GeneralInfoLogger')
general_logger.setLevel(logging.INFO)

# File handler for general information logger
general_handler = logging.FileHandler('general_info.log')
general_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add the file handler to the general information logger
general_logger.addHandler(general_handler)

# Configure the second logger for training information
training_logger = logging.getLogger('TrainingLogger')
training_logger.setLevel(logging.INFO)

# File handler for training information logger
training_handler = logging.FileHandler('training_info.log')
training_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add the file handler to the training information logger
training_logger.addHandler(training_handler)

# Example usage
general_logger.info("This is a general info log file.")
training_logger.info("This log file contains training-specific console messages.")


def display_training_config(train_loader):
    """
    Display and logs a summary of the training configuration and data.

    This function prints detailed information about the training setup, including the total number of epochs, the number
    of image samples processed per epoch, the total number of samples in the dataset, batch size, total number of
    batches (steps) during training, and the number of batches processed per epoch.
    :param train_loader: train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    """
    total_epochs = cfg.TRAIN.EPOCH
    images_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
    total_samples = len(train_loader.dataset)
    batch_size = cfg.TRAIN.BATCH_SIZE
    total_batches = len(train_loader)
    batches_per_epoch = total_batches // total_epochs

    info = (
        f"The total number of epochs is: {total_epochs}\n"
        f"The total number of individual image samples to process per epoch: {images_per_epoch}\n"
        f"The total number of individual samples to process during training: {total_samples}\n"
        f"The number of individual samples per batch: {batch_size}\n"
        f"The expected number of steps (batches) to take (process) during training: {total_batches}\n"
        f"The number of batches to process per epoch: {batches_per_epoch}"
    )
    print(info)
    training_logger.info(info)


def unfreeze_backbone(model):
    for layer in cfg.BACKBONE.TRAIN_LAYERS:
        for param in getattr(model.siam_subnet.resnet_siam_subnet.pretrained_resnet50, layer).parameters():
            param.requires_grad = True  # make parameter trainable (unfreeze)
        for m in getattr(model.siam_subnet.resnet_siam_subnet.pretrained_resnet50, layer).modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train()  # set the batch norm layer to training mode


def freeze_backbone(model):
    for param in model.siam_subnet.resnet_siam_subnet.pretrained_resnet50.parameters():
        param.requires_grad = False  # make parameter non-trainable (freeze)
    for m in model.siam_subnet.resnet_siam_subnet.pretrained_resnet50.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()  # set the batch norm layer to evaluation mode


def build_opt_lr(model, current_epoch=0):
    """
    Builds an optimizer and a learning rate based on the current epoch.
    The current epoch is also used to freeze or unfreeze relevant model parameters

    :param model: SiamCAR model being trained
    :param current_epoch: current training epoch

    :return: the optimizer and learning rate scheduler instance
    """
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:  # train the backbone if current epoch is >= cfg.BACKBONE.TRAIN_EPOCH
        unfreeze_backbone(model)
    else:
        freeze_backbone(model)
    # except the resnet backbone, set all the model parameters as trainable = True
    # i.e.: SiameseSubnet().pw_conv_r_star and CarSubnet()
    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.siam_subnet.resnet_siam_subnet.pretrained_resnet50.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.siam_subnet.pw_conv_r_star.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.car_subnet.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    # instantiate the optimizer while indicating: 1) which parameters to optimize, 2) their learning rate
    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # instantiate the lr scheduler, taking into account the optimizer and total number of epochs
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    return optimizer, lr_scheduler


# deprecated
def get_cur_epoch(model_filename):
    match = re.search(r"StpEpoch_(\d+)_Loss", model_filename)
    if match:
        cur_epoch = int(match.group(1))
        print(cur_epoch)  # Output: 20
    else:
        print("cur_epoch not found in filename. Defaulted to cur_epoch = 0")
        cur_epoch = 0
    return cur_epoch


def load_train_config(model, train_config_path):
    try:
        # Load model state dictionary
        loaded_pt = torch.load(train_config_path)
        print(f"Loaded model from: {train_config_path}")
        model.load_state_dict(loaded_pt['model'])
        cur_epoch = loaded_pt['last_epoch']
        model.train()  # must be executed AFTER loading the parameters
        optimizer, lr_scheduler = build_opt_lr(model, cur_epoch)
        optimizer.load_state_dict(loaded_pt['optimizer'])
    except:
        cur_epoch = get_cur_epoch(train_config_path)
        loaded_pt = torch.load(train_config_path)
        model.load_state_dict(loaded_pt)
        model.train()  # must be executed AFTER loading the parameters
        optimizer, lr_scheduler = build_opt_lr(model, cur_epoch)

    return model, optimizer, lr_scheduler, cur_epoch


def estimate_training_time(model, optimizer, train_loader, loss_fn, steps=1):
    """
    Estimates the training time while taking into account the backbone state (frozen or unfrozen)
    :param train_loader: torch train loader
    :param steps: the number of steps to use for the time estimation
    :return: estimated time of arrival eta, in seconds.
    """

    total_epochs = max(0, cfg.TRAIN.EPOCH - cfg.TRAIN.START_EPOCH)
    if total_epochs >= cfg.BACKBONE.TRAIN_EPOCH:
        backbone_epochs = max(0, cfg.BACKBONE.TRAIN_EPOCH)
        car_only_epochs = max(0, total_epochs - backbone_epochs)
    else:
        backbone_epochs = total_epochs
        car_only_epochs = 0

    msg = (
        "-----------------------------------------------------------------\n"
        "Estimating the total training time given the configuration below:\n"
        f"Starting epoch: {cfg.TRAIN.START_EPOCH}\n"
        f"total epochs = 20 - starting epoch = {total_epochs}\n"
        "--\n"
        f"backbone_epochs: {backbone_epochs}\n"
        f"car_only_epochs: {car_only_epochs}\n"
        "-----------------------------------------------------------------"
    )
    print(msg)
    training_logger.info(msg)

    timer = TrainTimer(train_loader)

    data = iter(train_loader).__next__()

    def train_steps_time():
        time_per_batch_list = []
        for step in range(0, steps):
            time_per_batch_st = time.time()
            pred_dict = model(data['search'], data['template'])
            loss_dic = loss_fn(pred_dict, data['gt_bbox'])
            weighted_loss = loss_dic['weighted'].item()

            if is_valid_number(weighted_loss):
                optimizer.zero_grad()
                loss_dic['weighted'].backward()
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()
            time_per_batch = time.time() - time_per_batch_st
            timer.convert_time(time_per_batch, print_msg="Time per batch is: ")
            time_per_batch_list.append(time_per_batch)

        time_per_batch_arr = np.array(time_per_batch_list)
        return time_per_batch_arr.mean()

    unfreeze_backbone(model)
    time_per_batch_unfreezed = train_steps_time()

    freeze_backbone(model)
    time_per_batch_freezed = train_steps_time()

    batches_per_epoch = len(train_loader) // cfg.TRAIN.EPOCH

    time_per_epoch_unfreezed = batches_per_epoch * time_per_batch_unfreezed
    time_per_epoch_freezed = batches_per_epoch * time_per_batch_freezed

    training_eta = (backbone_epochs * time_per_epoch_unfreezed) + (car_only_epochs * time_per_epoch_freezed)
    converted_time = timer.convert_time(training_eta, print_msg="Training Estimated Time of Arrival (ETA) is: ")
    training_logger.info(converted_time)
    return training_eta


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


def train(train_loader, model_save_dir, train_config_path="", smr_writer=None):
    """
    Trains the SiamCAR model
    :param train_loader: torch data loader instance
    :param model_save_dir: directory where trained model(s) is (are) saved
    :param train_config_path: configuration path to resume training
    :param smr_writer: torch summary writer instance
    :return:
    """
    torch.autograd.set_detect_anomaly(True)

    model = SiamCarV1()  # initialize model architecture with random weights

    # Check if a model path is provided for loading
    is_valid_path = train_config_path and os.path.exists(train_config_path)
    if is_valid_path:
        model, optimizer, lr_scheduler, cur_epoch = load_train_config(model, train_config_path)
    else:
        # Train from scratch (no valid model path or empty string)
        msg = "Training from scratch..."
        print(msg)
        training_logger.info(msg)
        model.train()  # must be executed AFTER loading the parameters
        msg = f"The current epoch is set to cfg.TRAIN.START_EPOCH = {cfg.TRAIN.START_EPOCH}"
        print(msg)
        training_logger.info(msg)
        cur_epoch = cfg.TRAIN.START_EPOCH
        optimizer, lr_scheduler = build_opt_lr(model, cur_epoch)

    cur_lr = lr_scheduler.get_cur_lr()
    smr_writer.add_scalar("LR/Epoch", cur_lr, cur_epoch)
    smr_writer.flush()

    loss_fn = SiamCarLoss()
    halt = False
    batches_per_epoch = len(train_loader) // cfg.TRAIN.EPOCH

    display_training_config(train_loader)

    stop_epoch = cfg.TRAIN.EPOCH

    data = iter(train_loader).__next__()

    smr_writer.add_graph(model, (data['search'], data['template']), use_strict_trace=False)
    smr_writer.flush()

    timer = TrainTimer(train_loader)
    min_weighted_loss = 10e9

    training_eta = estimate_training_time(model, optimizer, train_loader, loss_fn, steps=3)

    print("Estimation Finished")

    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        # Note: processed_batches_num is a batch counter that automatically increments by 1 after the method __getitem__ in the
        # customized data class has been executed 'cfg.TRAIN.BATCH_SIZE times' (e.g., '32' times).

        time_batch_start = time.time()

        pred_dict = model(data['search'], data['template'])
        loss_dic = loss_fn(pred_dict, data['gt_bbox'])
        weighted_loss = loss_dic['weighted'].item()

        next_epoch_reached = (cur_epoch != batch_idx // batches_per_epoch + cfg.TRAIN.START_EPOCH)
        if next_epoch_reached:  # True once the next epoch is reached
            smr_writer.add_scalars("Losses/Epoch", loss_dic, cur_epoch)
            smr_writer.flush()
            cur_epoch = batch_idx // batches_per_epoch + cfg.TRAIN.START_EPOCH
            if cur_epoch == stop_epoch:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == cur_epoch:
                general_logger.info('Training of the modified resnet backbone has been started.')
                training_logger.info('Training of the modified resnet backbone has been started.')
                optimizer, lr_scheduler = build_opt_lr(model, cur_epoch)

            lr_scheduler.step(cur_epoch)  # indicate the current epoch to take it into account for computing the lr
            cur_lr = lr_scheduler.get_cur_lr()
            smr_writer.add_scalar("LR/Epoch", cur_lr, cur_epoch)
            smr_writer.flush()
            training_logger.info('epoch: {}'.format(cur_epoch + 1))
        msg = (
            f"Epoch {cur_epoch} out {cfg.TRAIN.EPOCH}\n"
            f"Extracting the {batch_idx + 1}-th batch\n"
            f"Current learning rate = {cur_lr}"
        )
        print(msg)
        training_logger.info(msg)

        if is_valid_number(weighted_loss):
            smr_writer.add_scalars("Losses/Step", loss_dic, batch_idx)
            smr_writer.flush()
            optimizer.zero_grad()
            loss_dic['weighted'].backward()
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

            if min_weighted_loss > weighted_loss:
                min_weighted_loss = weighted_loss
                save_model(model, model_save_dir, cur_epoch, loss_dic['weighted'], optimizer, file_name='MinLossModel')
                general_logger.info(
                    f"Min loss model 'MinLossModel.pt' saved at epoch {cur_epoch} with loss {weighted_loss}")
                training_logger.info(
                    f"Min loss model 'MinLossModel.pt' saved at epoch {cur_epoch} with loss {weighted_loss}")
        else:
            general_logger.info("Invalid loss value !!")
        msg = f"Epoch {cur_epoch}, Batch {batch_idx}, Loss: {loss_dic['weighted'].item()}"
        training_logger.info(msg)
        print(msg)

        if halt:
            break
        elapsed_time = time.time() - start_time
        estimated_time_remaining = training_eta - elapsed_time
        estimated_time_remaining_str = timer.convert_time(total_seconds=estimated_time_remaining, print_msg="Estimated traintime remaining")
        training_logger.info(estimated_time_remaining_str)
    print("--")
    actual_training_time = time.time() - start_time
    actual_training_time_str = timer.convert_time(total_seconds=actual_training_time,
                                                  print_msg="The training took exactly: ")
    print(actual_training_time_str)
    training_logger.info(actual_training_time_str)
    save_model(model, model_save_dir, cur_epoch, loss_dic['weighted'], optimizer)  # Save the model after training

    print("--")


def save_model(model, model_save_dir, cur_epoch, loss_value, optimizer, file_name=None):
    # Check if model_save_dir exists, create if not
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        msg = f"Created model save directory: {model_save_dir}"
        print(msg)
        training_logger.info(msg)

    # Count existing model files (.pt)
    models_count = 0
    for filename in os.listdir(model_save_dir):
        if filename.endswith(".pt"):
            models_count += 1

    # Save the model
    rounded_loss = round(loss_value.item(), 2)  # Round loss to 2 decimal places
    # model_filename = f"Model_{models_count}_{rounded_loss}.pt"

    # ModelNbr is the order number of the saved model within the same folder
    # StpEpoch: the epoch at which we stoped training the model
    # Loss: is the loss value at StpEpoch
    if file_name is None:
        file_name = f"ModelNbr_{models_count}_StpEpoch_{cur_epoch}_Loss_{rounded_loss}.pt"
    else:
        file_name = file_name + '.pt'
    train_config_path = os.path.join(model_save_dir, file_name)

    train_config = {'last_epoch': cur_epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
    torch.save(train_config, train_config_path)
    msg = f"Model and its training configuration have been saved to: {train_config_path}"
    print(msg)
    training_logger.info(msg)


def print_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            msg = (
                f"Layer Name: {name}\n"
                f"Gradients: {param.grad}"
            )
            print(msg)
            general_logger.info(msg)


def main():
    # build dataset loader
    train_sampler = None
    if cfg.TRAIN.LOG_DIR:
        smr_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
        general_logger.info(f"Summary writer has been initialized. Log directory at: {cfg.TRAIN.LOG_DIR}")
    else:
        general_logger.info("No summary writer has been created. Missing directory cfg.TRAIN.LOG_DIR")
        smr_writer = None

    general_logger.info("Initializing the training dataset")
    train_dataset = TrkDataset(norm=False)
    general_logger.info("Dataset Initialized")
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)

    load_model = ''  # empty string will have the effect of training from scratch
    # load_model = '.\SavedModels\ModelNbr_4_StpEpoch_20_Loss_2.35.pt'
    train(train_loader, model_save_dir="SavedModels", train_config_path=load_model, smr_writer=smr_writer)
    smr_writer.flush()
    smr_writer.close()


if __name__ == '__main__':
    script_name = os.path.basename(__file__)
    msg = f"Script {script_name} has started the execution"
    print(msg)
    training_logger.info(msg)
    main()
