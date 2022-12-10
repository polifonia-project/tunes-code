import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
print(__file__, datetime.now().strftime("%H:%M:%S"))

#for repeated pitches:
#spans would be better for ixs [(start,end),(start,end),...]
def getColoredIxs(target_ixs, ixs, songlength):
    colorixs = []
    for ix in target_ixs:
        colorixs.append(ix)
        temp_ix = ix+1
        while (not temp_ix in ixs) and (temp_ix < songlength):
            colorixs.append(temp_ix)
            temp_ix += 1
    return colorixs

def array2colordict(array, ixs, criterion, songlength, color='red', greynan=True):
    ixsnp = np.array(ixs)
    target_ixs = ixsnp[np.where(criterion(array))]
    nan_ixs = []
    if greynan:
        nan_ixs = ixsnp[np.where(np.isnan(array))]        
    color_ixs = getColoredIxs(target_ixs, ixs, songlength)
    return {color:color_ixs, 'grey':nan_ixs}

#color all notes with high novelty
def novelty2colordict(novelty, ixs, percentile, songlength, color='red', greynan=True):
    criterion = lambda x : x >= np.nanpercentile(novelty,percentile)
    return array2colordict(
        novelty,
        ixs,
        criterion,
        songlength,
        color,
        greynan
    )

#color all notes with low consonance
def consonance2colordict(consonance, ixs, percentile, songlength, color='red', greynan=True):
    criterion = lambda x : x < np.nanpercentile(consonance,percentile)
    return array2colordict(
        consonance,
        ixs,
        criterion,
        songlength,
        color,
        greynan
    )

def plotArray(array, ixs, xlabel : str, ylabel : str):
    fig = plt.figure(figsize=(14,5))
    plt.ylim(0,np.max(np.nan_to_num(array)) * 1.05)
    plt.plot(array)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(0, len(ixs), 1), [str(i) for i in ixs])
    return fig