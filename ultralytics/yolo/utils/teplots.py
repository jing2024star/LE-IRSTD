# Ultralytics YOLO 🚀, AGPL-3.0 license
import os.path
import yaml
from ultralytics.yolo.utils.configfile import configureTrainFile,configureTestFile,mode
pathroot=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
pathdatasets=os.path.join(pathroot,"datasets")
pathcfg=os.path.join(pathdatasets,"cococfg.yaml")
with open(pathcfg,'r',encoding='utf-8') as f1:
    data = f1.read()
    result = yaml.load(data,Loader=yaml.FullLoader)
cfgmodel = result["cfgmodel"]
cfgexp = result["cfgexp"]

isshowpftrue=False
def level_map5095(digital=0):
    level_map5095 = [0,0,0,0,0,0,0,0,0,0,0]
    return level_map5095[digital]
def level(digital=0):
    lel = [0,0,0,0,0,0,0,0,0,0,0]
    return lel[digital]
def levelap50(digital=0):
    lap50 = [0,0,0,0,0,0,0,0,0,0,0]
    return lap50[digital]
def levelf1(digital=0):
    pass
def configmodel(config_string=0):
    pass
def configexp(exp=0):
    pass
def calculate_pr_curve():
    pass
def parse_knownss(mp=None,mr=None,map50=None,map=None):
    pass
def __(p=None, r=None, ap50=None, ap=None):#single
    pass
def calculateAP50(apap501=None):
    pass
def calculateprap(pp1=None,rr1=None,apap501=None,apap1=None):
    pass
def calculateF1(f1=None):
    pass
def speedcalculate(i=None,j=None, k=None, weights=None):
    pass