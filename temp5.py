import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.stats
from scipy.stats import sem
import statistics
from sklearn.metrics import mean_squared_error




#data = [0.1037982654406425,0.08235821821771828,3.672754812583681,2.218398376984698,0.26202540021473336, 0.15536200158992533, 0.702574044231008, 3.28868559961240337, 0.22810298186385536, 2.7706755302170519]

#data = [0.438511869949613, 0.07034783658873263, 0.06904238461175866, 0.03727939082312796, 0.8554829861107921, 0.01765315005742568, 0.24707514786004822, 0.05685977754351137, 0.04299540358021825, 0.07906425530567454]

#data = [0.46929714506356246, 0.6278421608134925, 0.18569916724937324, 0.48351134136984897, 0.708743985556633]

data = [0.2816157483525883, 0.10818076243601384, 0.09521358281844375, 0.15367692057567664, 0.24280102379584648]


print(data)

standard_error = sem (data)

print(standard_error)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    print(m, m-h, m+h, h, se)


mean_confidence_interval(data, confidence=0.95)


sd = np.std(data)
print("Population standard deviation of the dataset is", sd)



sd = statistics.pstdev(data)
print("Standard Deviation of the dataset is", sd)
