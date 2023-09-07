from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        
        self.writer = SummaryWriter(log_dir)

    def list_of_scalars_summary(self, tag_value_dictionary, step):
        
        for tag, value in tag_value_dictionary.items():
            self.writer.add_scalar(tag, value, global_step=step)

