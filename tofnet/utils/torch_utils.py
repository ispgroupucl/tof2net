import torch

def collate_batch(batch):
    """ Combines different samples in order to form a batch

    Arguments:
        batch (list): list of dictionaries

    Returns:
        dict where each sample has been concatenated by type
    """
    result = {}
    for key in batch[0].keys():
        values = (elem[key] for elem in batch)
        values = tuple(values)
        result[key] = torch.cat(values)
    return result
