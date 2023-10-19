ROOT = "data"
ORGAN = "spleen"

@dataclass
class CFG:
    # dataset
    path_to_images: str = f"{ROOT}/{ORGAN}_dataset/{ORGAN}_crops"
    path_to_meta_csv: str = f"{ROOT}/train_{ORGAN}.csv"
    fold: int = 4
    shape: tuple = (128, 128, 64) # whd
    target: str = 'multi'
    num_workes: int = 6
    group: str = "Egor"
    description: str = '''baseline adamw+plateuo only 1 scan weighted loss'''
    
    # training
    num_classes: int = 3
    batch_size: int = 8
    wandb_project: str = 'RSNA_classification_{ORGAN}'
    default_root_dir: str = 'cls'
    checkpoints_dir: str = 'cls/checkpoints'
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 300