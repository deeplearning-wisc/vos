import argparse
import contextlib
import cv2
import joblib
import numpy as np
import os
import random


from joblib import Parallel, delayed
from multiprocessing import Manager, cpu_count
from time import sleep
from tqdm import tqdm

# Project imports
from probabilistic_inference.inference_utils import corrupt

# Fix random seeds
np.random.seed(0)
random.seed(0)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class Counter(object):
    def __init__(self, manager, initval=0):
        self.val = manager.Value('i', initval)
        self.lock = manager.Lock()

    def reset(self, hard=False):
        with self.lock:
            if hard:
                self.val.value = 0
            elif self.val.value > 18:
                self.val.value = 0

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


def main(args):
    #########################################################
    # Specify Source Folders and Parameters For Frame Reader
    #########################################################
    dataset_dir = args.dataset_dir

    image_dir = os.path.expanduser(os.path.join(dataset_dir, 'val2017'))
    image_list = os.listdir(image_dir)

    max_corruption_levels = [1, 2, 3, 4, 5]
    # To get deterministic results across runs, keep this value 1. For faster dataset generation, uncomment cpu_count().
    num_cores = 1
    #num_cores = cpu_count()

    corruption_number = Counter(Manager(), initval=0)

    for corruption_level in max_corruption_levels:
        output_dir = os.path.expanduser(
            os.path.join(dataset_dir, 'val2017_' + str(corruption_level)))
        os.makedirs(output_dir, exist_ok=True)

        print(
            'Generating corrupted data at corruption level ' +
            str(corruption_level))

        with tqdm_joblib(tqdm(desc="Images corrupted:", total=len(image_list))) as _:
            Parallel(
                n_jobs=num_cores,
                backend='loky')(
                delayed(generate_corrupted_data)(
                    image_dir,
                    output_dir,
                    image_i,
                    corruption_level,
                    corruption_number) for image_i in image_list)

            corruption_number.reset(hard=True)


def generate_corrupted_data(
        image_dir,
        output_dir,
        image_i,
        corruption_level,
        corruption_number):

    image_tensor = cv2.imread(os.path.join(image_dir, image_i))
    image_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB)

    corruption_number.reset()
    corrupt_im = corrupt(
        image_tensor,
        severity=corruption_level,
        corruption_name=None,
        corruption_number=corruption_number.value())

    image_tensor = cv2.cvtColor(corrupt_im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, image_i), image_tensor)

    corruption_number.increment()


if __name__ == "__main__":
    # Create arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=str,
        help='bdd100k dataset directory')

    parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        help='converted dataset write directory')

    args = parser.parse_args()
    main(args)
