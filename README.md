# Decentralized FL for Foundation Model

This repository provides a **framework for distributed optimization of foundation models** with support for **centralized and decentralized FL**, **parametric and non-parametric aggregation**.

---

## Features

- **Federated Learning Settings**
  - Centralized Federated Learning
  - Decentralized Federated Learning (p2p, no server)

- **Aggregation Methods**
  - Parametric aggregation
  - Non-parametric aggregation

- **Communication Topologies ( For P2P)**
  - Ring
  - Regular graphs (configurable degrees)
  - Star
  - Random graphs

- **Model Support**
  - Prompt-tuned ViT
  - Prompt-tuned ViT with Prompt Pool (L2P)

- **More to be addied**

---

## Project Structure

    ├── config                           # wandb configs 
    ├── data                             # Put datasets here 
    ├── notebook                         # Jupyter Notebook
    ├── src
    │   ├── algo                         # Federated learning algorithms & communication
    │   │   ├── algo_factory.py         # Algorithm Factory   
    │   │   ├── communication.py         # Client–client / client–server communication logic
    │   │   ├── fl_base.py               # Base FL functions
    │   │   ├── fedavg.py                # Centralized FedAvg
    │   │   ├── decen_fedavg.py           # Decentralized FedAvg
    │   │   ├── nonparametric_aggregation.py  # Non-parametric aggregation
    │   │   └── topo.py  # Decentralized topology
    │   │
    │   ├── data_loader                  # Dataset loading and distribution
    │   │   ├── DataDistributer.py        # Data partitioning across clients
    │   │   ├── dataset_factory.py        # Dataset factory
    │   │   ├── Four_dataset_reader.py    # Fourdataset dataset reader
    │   │   ├── PACS_reader.py            # PACS dataset reader
    │   │   └── TinyImageNet_reader.py    # TinyImageNet dataset reader
    │   │
    │   ├── models                       # Model definitions
    │   │   ├── L2P.py                   # Learning-to-Prompt (L2P) model
    │   │   ├── regular_prompt.py         # Standard prompt-based model
    │   │   └── model_factory.py          # Model factory
    │   │
    │   ├── training                     # Training and evaluation loops
    │   │   ├── train.py                 # training logic
    │   │   ├── train_loop.py            # Core training loop
    │   │   └── eval.py                  # Evaluation logic
    │   │
    │   ├── util                         # Utility modules
    │   │    ├── argument_parser.py        # CLI argument parsing
    │   │    ├── constant.py               # Global constants
    │   │    └── misc.py                   # Miscellaneous helpers
    │   │   
    │   └── fl_training.py               # Main FL training file
    │
    ├── wandb                            # Weights & Biases logs
    │
    ├── .gitignore
    ├── environment.yml                  # Conda environment specification
    └── README.md


---

## Installation

Create and activate a virtual environment, then install dependencies:

    conda env create -f environment.yml
    conda activate torch

---
## Main Flow

For the main execution flow, see:

- `src/fl_training.py`

## Parameters

For detailed parameters and configuration options, see:

- `src/util/argument_parser.py`
---
## Sample Run with wandb

### Launch Sweeps

Centralized Federated Learning:

    wandb sweep config/sweep_cfl_para.yaml

Decentralized Federated Learning:

    wandb sweep config/sweep_dfl_para.yaml

### Run Sweep Agent

After creating a sweep, start an agent using the sweep ID:

    wandb agent <entity>/<project>/<sweep_id>
---
