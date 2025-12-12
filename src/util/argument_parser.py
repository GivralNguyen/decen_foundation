import argparse

def read_option():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--verbose', help='print out arg values;', action='store_true', default=False)
    
    # GENERAL SETTINGS 
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='[cuda | cpu]')
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    
    
    # FL SETTINGS
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset to choose: [cifar10 | cifar100 | PACS | tinyimagenet | fourdataset | cinic10 | mnistm | fashion | mmafedb]')
    parser.add_argument('--fl_setting', type=str, default='decentralized_fl',
                        help='Algorithm to choose: [centralized_fl | decentralized_fl]')
    parser.add_argument('--aggregation', type=str, default='parametric',
                        help='Algorithm to choose: [parametric | nonparametric]')
    parser.add_argument('--batch', type=int, default=16, 
                        help='batch size')
    parser.add_argument('--comm_rounds', type=int, default=120, 
                        help='communication rounds')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--local_eps', type=int, default=5,
                        help='number of epochs in local clients training')
    parser.add_argument('--n_clients', type=int, default=20,
                        help='the number of clients')
    parser.add_argument('--n_sampled_clients', type=int, default=20,
                        help='the number of sampled clients per round')
    
    # ADDITIONAL DECEN FL SETTINGS
    
    parser.add_argument('--topo', type=str, default='ring',
                        help='choose decen topo you want: [ring | regular | random | star | exponential | double_ring]')
    parser.add_argument('--degree', type=int, default='2',
                        help='degree for regular graph')
    
    # Data distribution
    parser.add_argument('--data_distribution', type=str, default='non_iid_dirichlet',
                        help='data split way to choose: [non_iid_dirichlet | manual_extreme_heterogeneity]')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='the level of non-iid data split')
    parser.add_argument('--n_dominated_class', type=int, default=1,
                        help='number of dominated class when applying manual_heterogeneity')
    
    # PROMPT TUNING SETTINGS
    parser.add_argument('--model_type', type=str, default='L2P',
                        help='choose model type you want: [prompted | L2P]')
    parser.add_argument('--prompt_method', type=str, default='shallow',
                        help='[shallow | deep], which layer to insert the prompt')
    parser.add_argument('--pool_size', type=int, default=20,
                        help='Define the prompt pool size in L2P model')
    parser.add_argument('--n_tokens', type=int, default=10,
                        help='number of prompts selected in L2_')
    parser.add_argument('--batchwise_prompt', type=bool, default=True,
                        help='Define L2P heuristic model if selecting top_k prompts in pool by batchwise')
    parser.add_argument('--reduce_sim_scalar', type=float, default=0.01,
                        help='control similarity between query and prompt in L2P model')
    
    # FED NON PARA SETTINGS
    parser.add_argument('--nonpara_hidden_size', type=int, default=128,
                        help='Define the number of hidden neurons in Nonparametric aggregation method')
    parser.add_argument('--num_loops_fednonpara', help='Number of outer loop in Probabilistic prompts;', type=int, default=5)
    
    # Wandb Settings
    parser.add_argument('--wandb', help='print out arg values;', action='store_true', default=False)
    
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option 