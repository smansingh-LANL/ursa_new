"""
ursa_new package initializer

Provides convenient imports for core utilities so downstream code can write:

    from ursa_new import (
        get_loaders,
        evaluate_massconservation_per_trajectory,
        get_train_val_test_splits,
        NetCDFTrajectoryDataset,
        close_all_datasets,
    )
"""

from ursa_new.metrics_eval import (
    get_loaders,
    evaluate_massconservation_per_trajectory,
)

from ursa_new.dataloader import (
    get_train_val_test_splits,
    NetCDFTrajectoryDataset,
    close_all_datasets,
)

__all__ = [
    # metrics_eval
    "get_loaders",
    "evaluate_massconservation_per_trajectory",
    # dataloader
    "get_train_val_test_splits",
    "NetCDFTrajectoryDataset",
    "close_all_datasets",
]
