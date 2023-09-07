import os
import torch

from lib.utils import load_class_names
from datasets.shipdataset import ShipDataset


def load_data(data_dir, dataset, action, img_size=608, sample_size=600, batch_size=4, shuffle=True, augment=True, mosaic=True, multiscale=True):
    class_names = load_class_names(os.path.join(data_dir, "class.names"))
    data_dir = os.path.join(data_dir, action)

    if dataset == "dataship-jpg":
        dataset = ShipDataset(data_dir, img_size=img_size, augment=augment, sample_size=sample_size, mosaic=mosaic, multiscale=multiscale)
    
    else: 
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   pin_memory=True, collate_fn=dataset.collate_fn)

    return dataset, dataloader
