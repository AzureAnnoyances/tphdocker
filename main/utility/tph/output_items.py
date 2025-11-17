import numpy as np
class Oitems:
    def __init__(self, i, xy, z_ffb, z_grd,h):
        self.i:int=i
        self.xy:np.ndarray=xy
        self.x = xy[0]
        self.y = xy[1]
        self.z_ffb:float=z_ffb
        self.z_grd:float=z_grd
        self.detected_trunk:bool=False
        self.detected_crown:bool=False
        self.h:float=h
        self.diam:float=0.0
        pass
    
    def asdict(self):
        return self.__dict__
    