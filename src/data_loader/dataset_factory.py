import torchvision.transforms as transforms
import torchvision
import torch.utils.data as Data
import numpy as np
from util.constant import alpha_seed_map, norm_stats, resize, centercrop_size
from data_loader.DataDistributer import DataPartitioner
from data_loader.TinyImageNet_reader import TinyImageNet_reader
from data_loader.PACS_reader import Pacs_reader
from data_loader.Four_dataset_reader import four_dataset_reader, Cinic10_reader,MNISTM,Fashion_RGB_reader,MMAFEDB_reader

def build_data_transform(normalize_stats, resize, centercrop_size,
                         interpolation_type=transforms.InterpolationMode.BILINEAR):
    transform = transforms.Compose([transforms.Resize(resize, interpolation=interpolation_type),
                                    transforms.CenterCrop(size=centercrop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*normalize_stats,inplace=True)])
    return transform


class DatasetFactory:
    # Dataset + preprocessing factory for FL experiments:
    # builds torchvision/custom datasets, applies a standard transform pipeline,
    # partitions train data across clients (Dirichlet non-IID or manual extreme heterogeneity),
    # and returns per-client train loaders, test loader(s), client weights, num_classes, and extra metadata.
    def __init__(self, _norm_stats = None, _resize = None, _centercrop_size = None):
        ns = norm_stats if _norm_stats is None else _norm_stats
        rs = resize if _resize is None else _resize
        cs = centercrop_size if _centercrop_size is None else _centercrop_size
        self.preprocess = build_data_transform(ns, rs, cs)
        self._registry = {}
        self._register_defaults()
        
    def build(self, name):
        name = name.lower()
        if name not in self._registry:
            raise ValueError(f"Dataset '{name}' is not supported")

        builder = self._registry[name]
        return builder(self.preprocess)
    
    def register(self, name):
        def _decorator(fn):
            key = name.lower()
            if key in self._registry:
                raise ValueError(f"Dataset '{name}' already registered")
            self._registry[key] = fn
            return fn

        return _decorator
    
    def prepare(self, args):
        print(f"[DatasetFactory] dataset = {args['dataset']}, "
          f"num_clients = {args['n_clients']}, "
          f"num_sampled_per_round = {args['n_sampled_clients']}, "
          f"batch_size = {args['batch']}")
        trainset, testset, num_classes, extra = self.build(args['dataset'])
         # Build test loaders
        testloaders = self._build_test_loaders(
            testset,
            batch_size=args['batch'],
            num_workers=2
        )
        data_partitioner = self._build_partitioners(
            trainset,
            n_clients=args['n_clients'],
        )
        if args['data_distribution'] == 'non_iid_dirichlet':
            print(f"[DatasetFactory] non_iid_dirichlet, "
            f"alpha = {args['alpha']} "
            )
            seed = alpha_seed_map.get(args['alpha'])
            if isinstance(data_partitioner, list):
                for i in range(len(data_partitioner)):
                    if i==0:
                        data_partitioner[i].dirichlet_split_noniid(alpha=args['alpha'], least_samples=32, manual_seed=seed)
                    else:
                        data_partitioner[i].dirichlet_split_noniid(alpha=args['alpha'], least_samples=32, manual_seed=data_partitioner[0].seed)
            else:
                data_partitioner.dirichlet_split_noniid(alpha=args['alpha'], least_samples=32, manual_seed=seed)
        elif args['data_distribution'] == 'manual_extreme_heterogeneity':
            print("[DatasetFactory] manual_extreme_heterogeneity, "
            )
            if isinstance(data_partitioner, list):
                for i in range(len(data_partitioner)):
                    data_partitioner[i].manual_allocating_noniid(args['n_dominated_class'], 0.99, 1.0)
            else:
                data_partitioner.manual_allocating_noniid(args['n_dominated_class'], 0.99, 1.0)
        else:
            raise ValueError("Input data distribution is not supported")
        
        partitioners = data_partitioner if isinstance(data_partitioner, list) else [data_partitioner]

        # weights: concat per-partitioner client weights
        all_clients_weights = np.concatenate([p.get_all_client_weights() for p in partitioners])

        # train loaders: flatten per-partitioner loader lists
        distributed_trainloaders = [
            loader
            for p in partitioners
            for loader in p.get_distributed_data(batch_size=args["batch"])
        ]

        return (
            distributed_trainloaders, # list[DataLoader] (one per client, flat)
            testloaders,              # DataLoader or list[DataLoader]
            all_clients_weights,      # np.ndarray, aligned with distributed_dataloaders
            num_classes,
            extra
        )
    
    def _build_test_loaders(self, testset, batch_size, num_workers):
        if isinstance(testset, (list, tuple)):
            return [
                Data.DataLoader(ds, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
                for ds in testset
            ]
        else:
            return Data.DataLoader(
                testset, batch_size=batch_size,
                num_workers=num_workers, shuffle=False
            )
    
    def _build_partitioners(self, trainset, n_clients):
        if isinstance(trainset, (list, tuple)):
            n_domains = len(trainset)
            clients_per_domain = n_clients // n_domains
            return [
                DataPartitioner(trainset[i], clients_per_domain)
                for i in range(n_domains)
            ]
        else:
            return DataPartitioner(trainset, n_clients)
    
    def _register_defaults(self):
        # CIFAR-10
        @self.register("cifar10")
        def _cifar10(preprocess):
            trainset = torchvision.datasets.CIFAR10(
                root="data/", train=True, download=True, transform=preprocess
            )
            testset = torchvision.datasets.CIFAR10(
                root="data/", train=False, download=True, transform=preprocess
            )
            num_classes = 10
            return trainset, testset, num_classes, {}

        # CIFAR-100
        @self.register("cifar100")
        def _cifar100(preprocess):
            trainset = torchvision.datasets.CIFAR100(
                root="data/", train=True, download=True, transform=preprocess
            )
            testset = torchvision.datasets.CIFAR100(
                root="data/", train=False, download=True, transform=preprocess
            )
            num_classes = 100
            return trainset, testset, num_classes, {}

        # PACS
        @self.register("pacs")
        def _pacs(preprocess):
            domains = ["art_painting", "cartoon", "photo", "sketch"]
            trainset = [
                Pacs_reader(
                    "../data/PACS/PACS",
                    split=d,
                    train=True,
                    transform=preprocess,
                    random_seed=i,
                )
                for i, d in enumerate(domains)
            ]
            testsets = [
                Pacs_reader(
                    "../data/PACS/PACS",
                    split=d,
                    train=False,
                    transform=preprocess,
                    random_seed=i,
                )
                for i, d in enumerate(domains)
            ]
            testset = Data.ConcatDataset(testsets)
            num_classes = 7
            extra = {"domains": domains}
            return trainset, testset, num_classes, extra

        # TinyImageNet
        @self.register("tinyimagenet")
        def _tinyimagenet(preprocess):
            trainset = TinyImageNet_reader(
                "data/tiny-imagenet-200/", train=True, transform=preprocess
            )
            testset = TinyImageNet_reader(
                "data/tiny-imagenet-200/", train=False, transform=preprocess
            )
            num_classes = 200
            return trainset, testset, num_classes, {}

        # FourDataset
        @self.register("fourdataset")
        def _fourdataset(preprocess):
            trainset, class_mask = four_dataset_reader(
                [30000] * 4, train=True, transform=preprocess
            )
            testset, _ = four_dataset_reader(
                [2500] * 4, train=False, transform=preprocess
            )
            num_classes = 37
            extra = {"class_mask": class_mask}
            return trainset, testset, num_classes, extra

        # Cinic10
        @self.register("cinic10")
        def _cinic10(preprocess):
            trainset = Cinic10_reader(
                root="data/cinic10-py",
                subset_size=30000,
                train=True,
                transform=preprocess,
            )
            testset = Cinic10_reader(
                root=".data/cinic10-py",
                subset_size=2500,
                train=False,
                transform=preprocess,
            )
            num_classes = 10
            return trainset, testset, num_classes, {}

        # MNISTM
        @self.register("mnistm")
        def _mnistm(preprocess):
            trainset = MNISTM(
                root="data/MNISTM",
                subset_size=30000,
                train=True,
                transform=preprocess,
            )
            testset = MNISTM(
                root="data/MNISTM",
                subset_size=2500,
                train=False,
                transform=preprocess,
            )
            num_classes = 10
            return trainset, testset, num_classes, {}

        # Fashion-RGB
        @self.register("fashion")
        def _fashion(preprocess):
            trainset = Fashion_RGB_reader(
                root="data/FashionMNIST",
                subset_size=30000,
                train=True,
                transform=preprocess,
            )
            testset = Fashion_RGB_reader(
                root="data/FashionMNIST",
                subset_size=2500,
                train=False,
                transform=preprocess,
            )
            num_classes = 10
            return trainset, testset, num_classes, {}

        # MMAFEDB
        @self.register("mmafedb")
        def _mmafedb(preprocess):
            trainset = MMAFEDB_reader(
                root="data/MMAFEDB",
                subset_size=30000,
                train=True,
                transform=preprocess,
            )
            testset = MMAFEDB_reader(
                root="data/MMAFEDB",
                subset_size=2500,
                train=False,
                transform=preprocess,
            )
            num_classes = 10
            return trainset, testset, num_classes, {}