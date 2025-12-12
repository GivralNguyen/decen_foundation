# train_loop.py
import numpy as np
from tqdm import trange
import wandb 
def run_train_loop(*,algo, args , testloader):
    """
    Standard decentralized FL training loop.

    Steps per round:
      1) client_train
      2) algo.agg()
      3) logging
      4) global_eval_avg

    Returns:
      eval_loss_record, eval_acc_record
    """


    for comm_round in trange(args['comm_rounds']):
        # 1) Local training on each client
        algo.client_train(
            comm_round=comm_round,
            epochs=args['local_eps'],
            lr=args['lr'],
            reduce_sim_scalar=args['reduce_sim_scalar'],
        )

        # 2) Decentralized aggregation (neighbors only)
        algo.agg()

        # 4) Evaluation: average metrics over all clients
        eval_loss, eval_acc = algo.global_eval_avg(
            testloader
        )
        if args['wandb']:
            wandb.log({"eval_loss": eval_loss, "round": comm_round})
            wandb.log({"test_acc": eval_acc, "round": comm_round})

