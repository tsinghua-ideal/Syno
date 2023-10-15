import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser
from KAS import Explorer

if __name__ == '__main__':
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Sampler
    model, sampler = models.get_model(args, return_sampler=True)

    # Explorer
    explorer = Explorer(sampler)

    node = explorer.interactive()
    if node and node.is_final():
        print("Realizing the result {} from explorer...", node)
        kernel_loader = sampler.realize(model, node, "trial")
        model.load_kernel(kernel_loader, compile=args.compile, batch_size=args.batch_size, seq_len=args.gpt_seq_len)
