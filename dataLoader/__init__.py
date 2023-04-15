from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
# from .spec_llff import SPECLLFFDataset


dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
            #    'specllff': SPECLLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
                'own_data':YourOwnDataset}