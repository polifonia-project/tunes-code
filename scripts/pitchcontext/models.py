"""
Models using the weighted pitch context vector.
"""

import numpy as np
from numpy.linalg import norm
from numpy import inf
from matplotlib import pyplot as plt

from .pitchcontext import PitchContext
from .song import Song

from datetime import datetime
print(__file__, datetime.now().strftime("%H:%M:%S"))

#distances
def cosineSim(v1, v2):
    """Cosine Similarity of `v1` and `v2`, both 1D numpy arrays"""
    return np.dot(v1,v2)/(norm(v1)*norm(v2))

def normalizedCosineSim(v1, v2):
    """Cosine Similarity of `v1` and `v2`, both 1D numpy arrays. Scaled between 0 and 1."""
    return (1.0 + np.dot(v1,v2)/(norm(v1)*norm(v2))) / 2.0

def normalizedCosineDist(v1, v2):
    """One minus the normalized cosine similarity"""
    return 1.0-normalizedCosineSim(v1, v2)

def sseDist(v1, v2):
    """Sum of squared differences between `v1` and `v2`, both 1D numpy arrays."""
    return np.sum((v1-v2)**2)


#find out how dissonant a note is in its context
#consonant in base40:
# perfect prime : Dp = 0
# minor third : Dp = 11
# major third : Dp = 12
# perfect fourth : Dp = 17
# perfect fifth : Dp = 23
# minor sixth : Dp = 28
# major sixth : Dp = 29

def computeConsonance(
    song : Song, 
    wpc : PitchContext,
    consonants40 = [0, 11, 12, 17, 23, 28, 29]
):
    """
    Computes for each note the consonance of the note given its context.

    Parameters
    ----------
    song : Song
        An instance of the Song class.
    wpd : WeightedPitchContext
        An instance of the WeightedPitchContext class, containing a weighted pitch context vector for each note.
    consonants : list of ints
        Intervals in base40 pitch encoding that are considered consonant.
    
    Returns
    -------
    consonance_pre, consonance_post, consonance_context : numpy 1D arrays
        with a consonance level for each note.
    """
    song_length = len(wpc.ixs)

    consonants = np.zeros( (40,) )
    consonants[consonants40] = 1.0

    #store result
    consonance_pre = np.zeros( song_length )
    consonance_post = np.zeros( song_length )
    consonance_context = np.zeros( song_length )

    for ix, context in enumerate(wpc.pitchcontext): #go over the notes...

        pitch40 = song.mtcsong['features']['pitch40'][wpc.ixs[ix]]-1

        #make copy of context
        context = np.copy(context)

        #normalize contexts: sum of context is 1.0
        context[:40] = context[:40] / np.sum(context[:40])
        context[40:] = context[40:] / np.sum(context[40:])

        intervals_pre  = np.roll(context[:40], -pitch40)
        intervals_post = np.roll(context[40:], -pitch40)

        consonance_pre[ix] = sum(np.multiply(intervals_pre, consonants))
        consonance_post[ix] = sum(np.multiply(intervals_post, consonants))

    #normalize
    #consonance_pre = consonance_pre / np.sum(consonance_pre)
    #consonance_post = consonance_post / np.sum(consonance_post)
    consonance_context = (consonance_pre + consonance_post) * 0.5

    return consonance_pre, consonance_post, consonance_context


def computePrePostDistance(
    song : Song,
    wpc : PitchContext,
    vectorDist=normalizedCosineDist
):
    res = np.zeros( len(wpc.pitchcontext) )
    for ix in range(len(wpc.pitchcontext)):
        res[ix] = vectorDist(wpc.pitchcontext[ix,:40], wpc.pitchcontext[ix,40:])
    return res

def computeNovelty(
    song: Song,
    wpc : PitchContext,
):
    novelty = np.zeros( len(wpc.pitchcontext) )
    for ix in range(len(wpc.pitchcontext)):
        total = wpc.pitchcontext[ix,:40] + wpc.pitchcontext[ix,40:]
        new = wpc.pitchcontext[ix,40:]
        total[new==0] = 1
        perc  = new / total
        novelty[ix] = np.average(perc[perc>0])
    return novelty