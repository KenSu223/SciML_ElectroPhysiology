import logging
import os
from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader

from .pt_dataset import PTDataset
from .web_utils import download_from_zenodo_record

from neuralop.utils import get_project_root

logger = logging.Logger(logging.root.level)

class EPDataset(PTDataset):
    """
    DarcyDataset stores data generated according to Darcy's Law.
    Input is a coefficient function and outputs describe flow. 

    Data source: https://zenodo.org/records/10994262

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """
    def __init__(self,
                 root_dir: Union[Path, str],
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: int=[16,32],
                 encode_input: bool=False, 
                 encode_output: bool=True, 
                 encoding="channel-wise",
                 channel_dim=1,
                 subsampling_rate=None,
                 download: bool=True):

        """DarcyDataset

        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        """

        # convert root dir to Path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        super().__init__(root_dir=root_dir,
                       dataset_name="ep",
                       n_train=n_train,
                       n_tests=n_tests,
                       batch_size=batch_size,
                       test_batch_sizes=test_batch_sizes,
                       train_resolution=train_resolution,
                       test_resolutions=test_resolutions,
                       encode_input=encode_input,
                       encode_output=encode_output,
                       encoding=encoding,
                       channel_dim=channel_dim,
                       input_subsampling_rate=subsampling_rate,
                       output_subsampling_rate=subsampling_rate)
        
example_data_root = get_project_root() / "neuralop/data/datasets/data"
def load_ep(n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    data_root = example_data_root,
    test_resolutions=[32],
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,):

    dataset = EPDataset(root_dir = data_root,
                           n_train=n_train,
                           n_tests=n_tests,
                           batch_size=batch_size,
                           test_batch_sizes=test_batch_sizes,
                           train_resolution=32,
                           test_resolutions=test_resolutions,
                           encode_input=encode_input,
                           encode_output=encode_output,
                           channel_dim=channel_dim,
                           encoding=encoding,
                           download=False)
    
    # return dataloaders for backwards compat
    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False,)
    
    test_loaders = {}
    for res,test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                       batch_size=test_bsize,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       persistent_workers=False,)
    
    return train_loader, test_loaders, dataset.data_processor
    
# legacy pt Darcy Flow loader
# def load_darcy_pt(n_train,
#                   n_tests,
#                   batch_size,
#                   test_batch_sizes,
#                   data_root = "./neuralop/data/datasets/data",
#                   train_resolution=16,
#                   test_resolutions=[16, 32],
#                   encode_input=False,
#                   encode_output=True,
#                   encoding="channel-wise",
#                   channel_dim=1,):

#     dataset = DarcyDataset(root_dir = data_root,
#                            n_train=n_train,
#                            n_tests=n_tests,
#                            batch_size=batch_size,
#                            test_batch_sizes=test_batch_sizes,
#                            train_resolution=train_resolution,
#                            test_resolutions=test_resolutions,
#                            encode_input=encode_input,
#                            encode_output=encode_output,
#                            encoding=encoding,
#                            channel_dim=channel_dim,
#                            download=False)
    
#     # return dataloaders for backwards compat
#     train_loader = DataLoader(dataset.train_db,
#                               batch_size=batch_size,
#                               num_workers=0,
#                               pin_memory=True,
#                               persistent_workers=False,)
    
#     test_loaders = {}
#     for res,test_bsize in zip(test_resolutions, test_batch_sizes):
#         test_loaders[res] = DataLoader(dataset.test_dbs[res],
#                                        batch_size=test_bsize,
#                                        shuffle=False,
#                                        num_workers=0,
#                                        pin_memory=True,
#                                        persistent_workers=False,)
    
#     return train_loader, test_loaders, dataset.data_processor