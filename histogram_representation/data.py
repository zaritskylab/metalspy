import pandas as pd
import numpy as np
from typing import List

class Leap:
    
    def __init__(self, **kargs) -> None:
        assert (kargs['is_responded'] and (not kargs['is_extreme_responded'])) or (not kargs['is_responded'])
        self.id: str = kargs['id']
        self.is_responded: bool = kargs['is_responded'] if 'is_responded' in kargs else None
        self.is_extreme_responded: bool = kargs['is_extreme_responded'] if 'is_extreme_responded' in kargs else None
        self.biobank_id: str = kargs['biobank_id']
        self.is_core: bool = kargs['is_core']
        self.mask: np.ndarray = kargs['mask']
        self.image: np.ndarray = kargs['image']
        self.pixels: pd.DataFrame = kargs['pixels']
        self.file_tag_id: str = kargs['file_tag_id']
        self.file_uid: str = kargs['file_uid']
        self.chunk_id: str = kargs['chunk_id']
        self.tic: np.ndarray = kargs['tic']
        self.batch_date: str = kargs['batch_date']
        self.lod: np.ndarray = kargs['lod']
        self.medium: str = kargs['medium']
        self.is_force: str = kargs['is_force']
        self.analysis_id = None
        self.next: List[Leap] = []
