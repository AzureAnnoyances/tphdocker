class OutputFolder:
    def __init__(self, input_folder, output_folder, topViewOut, sideViewOut, pcdOut, diamOut, debugOut):
        self.input_folder:str = input_folder
        self.output_folder:str = output_folder
        self.topViewOut:str = topViewOut
        self.sideViewOut:str = sideViewOut
        self.pcdOut:str = pcdOut
        self.diamOut:str = diamOut
        self.debugOut:str = debugOut

class PreprocessFolder:
    def __init__(self, root, pre_diam, pre_crown, pre_pcd, post_diam, post_crown):
        self.root:str = root
        self.pre_diam:str = pre_diam
        self.pre_crown:str = pre_crown
        self.pre_pcd:str = pre_pcd
        self.post_diam:str = post_diam
        self.post_crown:str = post_crown
