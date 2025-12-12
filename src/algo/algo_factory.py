from algo.fedavg import fedavg
from algo.decen_fedavg import decenfedavg
class AlgoFactory:
    # Factory for init centralized or decentralized FedAvg variants
    # based on FL setting (centralized / decentralized), aggregation type
    # (parametric / nonparametric), and communication topology (decen only).
    @staticmethod
    def create(args, **kwargs):
        fl = args['fl_setting']
        agg = args['aggregation']
        if fl == 'decentralized_fl':
            if args['topo'] == 'regular':
                print(
                    f"[AlgoFactory] fl={fl} | agg={agg} | "
                    f"Graph={args['topo']} | degree={args['degree']}"
                )
            else:
                print(
                    f"[AlgoFactory] fl={fl} | agg={agg} | "
                    f"Graph={args['topo']}"
                )
        else:
            print(f"[AlgoFactory] fl={fl} | agg={agg}")
                  
        if fl == 'centralized_fl' and agg == 'parametric':
            return fedavg(aggregation_method=agg, **kwargs)

        elif fl == 'centralized_fl' and agg == 'nonparametric':
            return fedavg(aggregation_method=agg, **kwargs)

        elif fl == 'decentralized_fl' and agg == 'parametric':
            return decenfedavg(aggregation_method=agg, G= args['topo'], **kwargs)

        elif fl == 'decentralized_fl' and agg == 'nonparametric':
            return decenfedavg(aggregation_method=agg, G= args['topo'], **kwargs)

        else:
            raise ValueError(f"Invalid algo combo: fl={fl}, agg={agg}")
