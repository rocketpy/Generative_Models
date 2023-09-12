# MusicGen Trainer - This is a trainer for MusicGen model.

# https://github.com/chavinlo/musicgen_trainer


# To load them, simply run the following on your generation script:

# model.lm.load_state_dict(torch.load('models/lm_final.pt'))

# Where model is the MusicGen Object and models/lm_final.pt is the path to your model (or checkpoint).


# Running the trainer
"""
Run python3 run.py --dataset /home/ubuntu/dataset, replace /home/ubuntu/dataset with the path to your dataset.
Make sure to use the full path, not a relative path.
Options:
    dataset_path: String, path to your dataset with WAV and TXT pairs.
    model_id: String, MusicGen model to use. Can be small/medium/large. Default: small
    lr: Float, learning rate. Default: 0.0001/1e-4
    epochs: Integer, epoch count. Default: 5
    use_wandb: Integer, 1 to enable wandb, 0 to disable it. Default: 0 = Disabled
    save_step: Integer, amount of steps to save a checkpoint. Default: None
Set these options like this: python3 run.py --use_wandb=1
"""
