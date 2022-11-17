
from torch.utils.data import Dataset, DataLoader


from .data_utils import *



datasets = {}
def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make_dataset(name,split, cfg, is_training=True):
    print(name)
    return datasets[name](split=split, is_training=is_training, **cfg)


def make_data_loader(dataset, generator, batch_size, num_workers, is_training):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True,
    )