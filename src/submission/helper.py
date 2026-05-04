from matplotlib.style import context

from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig

import torch
import random
random.seed(0)

def initialize_vanilla_model(mconf):
    attention_model = None
    ### TODO:
    ### [part d]: Make some model here

    ### START CODE HERE
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model

def initialize_rope_model(mconf):
    attention_model = None
    ### TODO
    ### [part h]: Make some other model here

    ### START CODE HERE
    # Enable rope by setting the appropriate configuration parameters in mconf
    mconf.rope = True
    attention_model = GPT(mconf)
    ### END CODE HERE
    return attention_model

def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model, finetune_lr=6e-4, writer=None):
    ### TODO:
    ### [part d] [part f]:
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=0
    ###
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu'), weights_only=True) to load pretrained model 

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)
    
    ### START CODE HERE
    
    # Load the finetuning dataset from the specified path
    finetune_corpus = open(finetune_corpus_path, 'r').read()
    train_dataset = NameDataset(data = finetune_corpus, pretraining_dataset = pretrain_dataset)

    # For finetuning without a pretrained model, we can directly create a TrainerConfig object with the specified hyperparameters and then create a Trainer object using this configuration and the model.
    if reading_params_path is None and finetune_corpus_path is not None:
        tconf = TrainerConfig(
            max_epochs=75,
            batch_size=256,
            learning_rate=finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=0,
        )

        trainer_obj = Trainer(model, train_dataset = train_dataset, test_dataset = None, config = tconf)   

    # For finetuning with a pretrained model, we first need to load the pretrained parameters into the model, and then we can create the TrainerConfig and Trainer objects as before.
    else:
        # Load the pretrained model parameters from the specified path
        pretrained_state_dict = torch.load(reading_params_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(pretrained_state_dict)

        tconf = TrainerConfig(
            max_epochs=10,
            batch_size=256,
            learning_rate=finetune_lr,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=200*len(pretrain_dataset)*block_size,
            num_workers=0,
        )

        trainer_obj = Trainer(model, train_dataset = train_dataset, test_dataset = None, config = tconf)

    ### END CODE HERE
    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    ### TODO:
    ### [part f]:
    ### - Given:
    ###     1. A corpus specified in pretrain_dataset
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=0

    trainer_obj = None #Trainer object (see trainer.py for more details)
    tconf = None #TrainerConfig object (see trainer.py for more details)

    ### START CODE HERE
    tconf = TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=pretrain_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=200*len(pretrain_dataset)*block_size,
        num_workers=0,
    )

    trainer_obj = Trainer(model, train_dataset = pretrain_dataset, test_dataset = None, config = tconf)
    ### END CODE HERE
    return tconf, trainer_obj

def train(model, writing_params_path, trainer_obj):
    ### TODO:
    ### - Given:
    ###     An output path writing_params_path for the model parameters
    ### [part d]:
    ###
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    ### START CODE HERE
    trainer_obj.train()
    # After training is complete, save the model parameters to the specified path
    torch.save(model.state_dict(), writing_params_path)
    ### END CODE HERE
    return
