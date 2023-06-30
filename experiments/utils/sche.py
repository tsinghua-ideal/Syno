from timm.scheduler import create_scheduler


def get_schedule(optimizer, args):
    return create_scheduler(args, optimizer)
