
import os.path
import yaml
pathroot=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
pathdatasets=os.path.join(pathroot,"datasets")
pathcfg=os.path.join(pathdatasets,"cococfg.yaml")
with open(pathcfg,'r',encoding='utf-8') as f1:
    data = f1.read()
    result = yaml.load(data,Loader=yaml.FullLoader)
cfgmodel = result["cfgmodel"]
cfgexp = result["cfgexp"]
configureTrainFileString=""
configureTestFileString=""
modeString="train"
def configureTrainFile():
    pass
def configureTestFile():
    pass
def mode():
    pass
def __train(cfg1=None):
    pass
def __test(cfg2=None):
    pass
def __predict(cfg3=None):
    pass