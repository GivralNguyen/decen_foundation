from models.L2P import L2P_ViT_B32
from models.regular_prompt import Prompted_ViT_B32
class ModelFactory:
    @staticmethod
    def create(args, num_classes):
        model_name = f"{args['model_type']}_Vit_B32" # will add more
        print(
            f"[ModelFactory] model={model_name} | "
            f"n_tokens={args.get('n_tokens')} | "
            f"pool={args.get('pool_size')} | "
            f"batchwise={args.get('batchwise_prompt')} | "
            f"classes={num_classes}"
        )
        if args['model_type'] == 'prompted':
            
            return Prompted_ViT_B32(
                weight_init='random',
                prompt_method=args['prompt_method'],
                num_tokens=args['n_tokens'],
                num_classes=['num_classes']
            )

        elif args['model_type'] == 'L2P':
            return L2P_ViT_B32(
                prompt_method=args['prompt_method'],
                batchwise_prompt=args['batchwise_prompt'],
                pool_size=args['pool_size'],
                top_k=args['n_tokens'],
                num_classes=num_classes
            )

        else:
            raise ValueError(f"Unknown model_type: {args['model_type']}")
