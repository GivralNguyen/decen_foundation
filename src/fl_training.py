import torch
import torch.nn as nn
import wandb
from util import argument_parser
from util.misc import setup_seed, to_device
from data_loader.dataset_factory import DatasetFactory
from algo.fl_base import FL_base
from algo.algo_factory import AlgoFactory
from models.model_factory import ModelFactory
from training.train_loop import run_train_loop
#MAIN

def main():
    
    args = argument_parser.read_option()
    # if args['verbose']: # verbose prints out arguments
    #     print(args)
    if args['wandb']: # Wandb setup
        wandb.init()
    setup_seed(args['seed']) # Set randomized seed
    
    data_factory = DatasetFactory()
    distributed_trainloaders, testloaders, all_clients_weights, num_classes, extra = data_factory.prepare(args)
    # preprocess_transform = data_loader.dataset_builder.build_data_transform(norm_stats, resize, centercrop_size)
    fl_base = FL_base(all_clients_weights,args['n_clients'], args['n_sampled_clients'], distributed_trainloaders)
    
    base_model = to_device(ModelFactory.create(args, num_classes), args['device'])
    base_model.build_trainable_keys()
    
    algo = AlgoFactory.create(
    args=args,
    base_model=base_model,
    scenario=fl_base,
    loss_fun=nn.CrossEntropyLoss(),
    class_mask= extra.get("class_mask") if extra is not None else None,
    device=args['device']
    )
    
    run_train_loop(algo=algo,args=args,testloader=testloaders)
    
    # if args['fl_setting'] == "centralized_fl":
    #     if args['aggregation'] == "parametric":
        
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()