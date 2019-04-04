#!/usr/bin/env python
import os, datetime

import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory, print_log
from db.datasets import datasets

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True



def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def prefetch_data(db, queue, sample_data, data_aug, debug=False):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug, debug=debug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e
            # exit()

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()
        # if configs["cuda_flag"]:#yezheng  
        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        print("[train.py pin_memory VALUE] training_queue", training_queue.qsize(), 
        "pinned_training_queue", pinned_training_queue.qsize(), 
        "training_pin_semaphore", training_pin_semaphore._value)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn, data_aug, debug=False):
    #=======
    tasks = [torch.multiprocessing.Process(target=prefetch_data, 
                     args=(db, queue, fn, data_aug, debug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    #========
    return tasks

def train(training_dbs, validation_db, start_iter=0, debug=False):
    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    # val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    # validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = torch.multiprocessing.Queue(system_configs.prefetch_size)
    # validation_queue = torch.multiprocessing.Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size)
    # pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data)
    sample_data = importlib.import_module(data_file).sample_data

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(
        training_dbs, training_queue, sample_data, True, debug)
    # if val_iter:
    #     validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data, False)

    training_pin_semaphore   = threading.Semaphore()
    # validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()

    # validation_pin_semaphore.acquire()
    
    #-----------
    # print("[train.py VALUE] training_queue", training_queue.qsize(), 
    #     "pinned_training_queue", pinned_training_queue.qsize(), 
    #     "training_pin_semaphore", training_pin_semaphore._value)
    # [train.py VALUE] training_queue 0 pinned_training_queue 0 training_pin_semaphore 0
    #-----------
    # print("[train.py VALUE] training_queue", training_queue, 
    #     "pinned_training_queue", pinned_training_queue, 
    #     "training_pin_semaphore", training_pin_semaphore)
    #-----------
    # [train.py VALUE] training_queue <multiprocessing.queues.Queue object at 0x7f6fd53d8cf8> 
    # pinned_training_queue <queue.Queue object at 0x7f6fd5267390> 
    # training_pin_semaphore <threading.Semaphore object at 0x7f6fd5267470>
    
    # print("[train.py TYPE] training_queue", type(training_queue), 
    #     "pinned_training_queue", type(pinned_training_queue), 
    #     "training_pin_semaphore", type(training_pin_semaphore))
    #-----------
    # [train.py TYPE] training_queue <class 'multiprocessing.queues.Queue'> 
    # pinned_training_queue <class 'queue.Queue'> 
    # training_pin_semaphore <class 'threading.Semaphore'>
    #-----------
    if False:
        training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
        training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
        training_pin_thread.daemon = True
        training_pin_thread.start()
    #-----------
    # print("[train.py VALUE] training_queue", training_queue.qsize(), 
    #     "pinned_training_queue", pinned_training_queue.qsize(), 
    #     "training_pin_semaphore", training_pin_semaphore._value)
    # [train.py VALUE] training_queue 0 pinned_training_queue 0 training_pin_semaphore 0
    #-----------
    # validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    # validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    # validation_pin_thread.daemon = True
    # validation_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(training_dbs[0], configs["cuda_flag"])
    print("[train] pretrained_model", pretrained_model)
    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))

        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
        print_log("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate),
            system_configs)
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    if torch.cuda.is_available() and configs["cuda_flag"]:
        nnet.cuda()
    nnet.train_mode()
    avg_loss = AverageMeter()
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm.tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss = nnet.train(**training)
            avg_loss.update(training_loss.item())

            if display and iteration % display == 0:
                print("training loss at iteration {}: {:.6f} ({:.6f})".format(
                    iteration, training_loss.item(), avg_loss.avg))
                print_log("training loss at iteration {}: {:.6f} ({:.6f})".format(
                    iteration, training_loss.item(), avg_loss.avg),
                system_configs)
            del training_loss

            # if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
            #     nnet.eval_mode()
            #     validation = pinned_validation_queue.get(block=True)
            #     validation_loss = nnet.validate(**validation)
            #     print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
            #     nnet.train_mode()

            if iteration % snapshot == 0:
                nnet.save_params(iteration)

            if iteration % 1000 == 0:
                nnet.save_params(-1)
                avg_loss = AverageMeter()

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

    # sending signal to kill the thread
    print("[train.py VALUE (before relase())] training_queue", training_queue.qsize(), 
        "pinned_training_queue", pinned_training_queue.qsize(), 
        "training_pin_semaphore", training_pin_semaphore._value)
    training_pin_semaphore.release()
    print("[train.py VALUE (after relase())] training_queue", training_queue.qsize(), 
        "pinned_training_queue", pinned_training_queue.qsize(), 
        "training_pin_semaphore", training_pin_semaphore._value)
    # validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    # for validation_task in validation_tasks:
    #     validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()
    # print("[train] args.cfg_file", args.cfg_file)
    # [train] args.cfg_file ExtremeNet
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    print("current time:{}".format(datetime.datetime.now()))
    print_log("============================",system_configs)
    print_log("current time:{}".format(datetime.datetime.now()),system_configs)
    
    print("loading all datasets...")
    print_log("loading all datasets...", system_configs)
    dataset = system_configs.dataset
    # threads = max(torch.cuda.device_count() * 2, 4)
    threads = args.threads
    print("using {} threads".format(threads))
    print_log("using {} threads".format(threads), system_configs)
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    # print("[train] training_dbs", training_dbs)
    # Remove validation to save GPU resources
    # validation_db = datasets[dataset](configs["db"], val_split)

    print("system config...")
    print_log("system config...", system_configs)
    pprint.pprint(system_configs.full)
    print_log(str(system_configs.full),system_configs)

    print("db config...")
    print_log("db config...",system_configs)
    pprint.pprint(training_dbs[0].configs)
    print_log(str(training_dbs[0].configs ) , system_configs)

    print("len of db: {}".format(len(training_dbs[0].db_inds)))
    print_log("len of db: {}".format(len(training_dbs[0].db_inds)) , system_configs)
    # train(training_dbs, validation_db, args.start_iter)
    train(training_dbs, None, args.start_iter, args.debug)
