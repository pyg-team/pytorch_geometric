import sys

_wandb_initialized: bool = False


def init_wandb(name: str, **kwargs):
    if '--wandb' not in sys.argv:
        return

    from datetime import datetime

    import wandb

    wandb.init(
        project=name,
        entity='pytorch-geometric',
        name=datetime.now().strftime('%Y-%m-%d_%H:%M'),
        config=kwargs,
    )

    global _wandb_initialized
    _wandb_initialized = True


def log(**kwargs):
    print(', '.join(f'{k}: {v}' for k, v in kwargs.items()))

    if _wandb_initialized:
        import wandb
        wandb.log(kwargs)
