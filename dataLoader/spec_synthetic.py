from llff import LLFFDataset

class FAKEDataset(LLFFDataset):

    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8):
        super().__init__(datadir, split, downsample, is_stack, hold_every)

    
    def read_meta(self):
        
