# train_loop.py
import numpy as np
from tqdm import trange
import wandb 
def run_train_loop(*,algo, args , testloader):
    """
    Steps per round:
      1) clients_train
      2) aggregation
      3) Evaluation
    """


    for comm_round in trange(args['comm_rounds']):
        # 1) Local training on each client
        algo.client_train(
            comm_round=comm_round,
            epochs=args['local_eps'],
            lr=args['lr'],
            reduce_sim_scalar=args['reduce_sim_scalar'],
        )

        # 2) aggregation 
        algo.agg(comm_round=comm_round)

        # 3) Evaluation
        eval_loss, eval_acc = algo.global_eval_avg(
            testloader
        )
        # if args['wandb']:
            # print("logging to wandb")
        wandb.log({"eval_loss": eval_loss, "round": comm_round})
        wandb.log({"eval_acc": eval_acc, "round": comm_round})

