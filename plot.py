import numpy as np

def subplot_gridsize(num):
    """ stolen from Matt Johnson - much better than my version
    https://github.com/mattjj/pymattutil/plot.py"""
    return sorted(min([(x,int(np.ceil(num/x))) for x in range(1,int(np.floor(np.sqrt(num)))+1)],key=sum))
