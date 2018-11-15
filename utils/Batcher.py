

class BatchGenerator(object):
    def __init__(self, config, data):
        self.data = data
        self.config = config
        self.batch_size = config['batch_size']

    def clip_tail(self):
        return

    def normalize(self):
        return

    def __len__(self):
        return

    def __iter__(self):
        return