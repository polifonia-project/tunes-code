from fractions import Fraction
import numpy as np

from .song import Song
from .base40 import base40

#from datetime import datetime
#print(__file__, datetime.now().strftime("%H:%M:%S"))

#weighted pitch contect
class PitchContext:
    """Class for computing a weighted pitch context vector and keeping track of the parameters.
    
    Parameters
    ----------
    song : Song
        instance of the class Song, representing the song as MTCFeatures, with some
        additional features. By instanciating an object of this class, the following
        parameters have to be provided. Parameters with a default value can be ommitted.
    removeRepeats : boolean, default=True
        If True, skip all notes with repeated pitches.
    syncopes : boolean, default=False
        If True, take the highest metric weight DURING the span of the note.
        If False, take the metric weight at the onset time of the note.
    metric_weights : one of 'beatstrength', 'ima', 'imaspect', default='beatstrength'
        `beatstrength` : use beatstrength as computed by music21
        `ima` : use inner metric analysis weights (not implemented)
        `imaspect` : use inner metric analysis spectral weights (not implemented)
    accumulateWeight : boolean, default=False
        If true, represent the metric weight of a note by the sum of all metric weights
        in the beatstrength grid in the span of the note.
    len_context_beat : float of (float, float)
        length of the context in beat units, either float or tuple
        (length pre context, length post context) in beat units
        TODO: if len_context_beat='auto', extend context to first note with higher metric weight (or equal if weight = 1.0)
    use_metric_weights : boolean or (boolean, boolean), default=True
        Whether to weight the pitches in the conext by their metric weight.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
    use_distance_weights : boolean or (boolean, boolean), default=True
        If True, weight pithces in the context by their distance to the focus note.
        The weight is a linear function of the score distance to the focus note.
        The weight at the onset of the focus note is 1.0.
        The weight at the end of the context is set by `min_distance_weight`.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
    min_distance_weight : float of (float, float), default=0.0
        Distance weight at the border of the context.
        If a tuple is given, first in the tuple refers to preceding context and
        second in the tuple to the following context.
    includeFocus : One of 'none', 'pre', 'post', 'both', default='none'
        Whether to include the focus note in the context
    partialNotes : boolean, default=True
        If True, extend the PRE conext to the START of the first note within the context.
        This has consequences if the pre context starts IN a note.
    normalize : boolean, default=False
        If True, normalize the weighted pitch context vector such that the values add up to 1.0.
        epsilon : float, default=1e-4
        Used for floating point comparisons.

    Attributes
    ----------
    params : dict
        Parameters for computing the WPC.
    ixs : list
        List of the indices of the notes that can be part of a context. This list is the single
        point of conversion to actual note indices within the melody.
    weightedpitch : numpy array
        Dimension is (length of ixs, 40). The first dimension corresponds to the note indices in
        `ixs`. The second dimension contains the metric weight of the corresponding note for the
        appropriate pitch in base 40 encoding.
    pitchcontext : numpy array
        Dimension is (length of ixs, 80). The first dimension corresponds to the note indices in
        `ixs`. The second dimension correpsonds to 40 pitches in the preceding context [:40] and
        40 pitches in the following context [40:]. Pitches are in base40 encoding.
    contexts_pre : list of lists
        Length is length of isx. For each note in `ixs`, `context_pre` contains a list of the
        indices of pitches in ixs that are part of the preceding context of the note.
    contexts_post : list of lists
        Length is length of isx. For each note in `ixs`, `context_post` contains a list of the
        indices of pitches in ixs that are part of the following context of the note.
    """

    def __init__(self, song, **inparams):
        self.song = song
        #contains params for actual contents of weightedpitch vector and weightedpitch context vector
        self.params = self.createDefaultParams()
        self.setparams(inparams)
        self.weightedpitch, self.ixs = self.computeWeightedPitch()
        self.pitchcontext, self.contexts_pre, self.contexts_post = self.computePitchContext()

    def createDefaultParams(self):
        """Return a dictionary with default parameters.
        
        Returns
        ------
        dictionary
            A Dictionary with all parameters and default values:
            ```{
                'removeRepeats' : True,
                'syncopes' : False,
                'metric_weights' : 'beatstrength',
                'accumulateWeight' : False,
                'len_context_beat' : None,
                'use_metric_weights' : True,
                'use_distance_weights' : True,
                'min_distance_weight' : 0.0,
                'includeFocus' : 'none',
                'partialNotes' : True,
                'normalize' : False,
                'epsilon' : 1e-4,
            }```
        """

        params = {
            'removeRepeats' : True,
            'syncopes' : False,
            'metric_weights' : 'beatstrength',
            'accumulateWeight' : False,
            'len_context_beat' : None,
            'use_metric_weights' : True,
            'use_distance_weights' : True,
            'min_distance_weight' : 0.0,
            'includeFocus' : 'none',
            'partialNotes' : True,
            'normalize' : False,
            'epsilon' : 1e-4,
        }
        return params
    
    def setparams(self, params):
        """Set parameters in `params`

        Parameters
        ----------
        params : dict
            key value pairs for the parameters to change
        """
        for key in params.keys():
            if key not in self.params:
                print(f"Warning: Unused parameter: {key}")
            else:
                self.params[key] = params[key]

    def computeWeightedPitch(self):
        """Computes for every note a pitchvector (base40) with the (metric) weight of the note in the corresponding pitch bin.

        Returns
        -------
        numpy array
            Dimension is (length of ixs, 40). The first dimension corresponds to the note indices in
            `ixs`. The second dimension contains the metric weight of the corresponding note for the
            appropriate pitch in base 40 encoding.
        """
        #put param values in local variables for readibility
        removeRepeats = self.params['removeRepeats']
        syncopes = self.params['syncopes']
        metric_weights = self.params['metric_weights']
        accumulateWeight = self.params['accumulateWeight']

        songinstance = self.song
        song = self.song.mtcsong

        if metric_weights in ['ima', 'imaspect']:
            raise Exception(f'{metric_weights} not yet implemented.')
        
        onsettick = song['features']['onsettick']
        pitch40 = song['features']['pitch40']
        beatstrengthgrid = np.array(song['features']['beatstrengthgrid'])
        beatstrength = song['features']['beatstrength']

        song_length = songinstance.getSongLength()
        ixs = []
        if removeRepeats:
            p_prev=-1
            for ix, p40 in enumerate(song['features']['pitch40']):
                if p40 != p_prev:
                    ixs.append(ix)
                p_prev = p40
        else:
            ixs = list(range(song_length))

        weights = [0]*len(ixs)

        if accumulateWeight:
            if syncopes:
                syncopes=False
                print("Warning: setting accumulateWeight implies syncopes=False.")
            max_onset = len(beatstrengthgrid)-1
            #for each note make span of onsets:
            start_onsets = [onsettick[ix] for ix in ixs]
            stop_onsets = [onsettick[ix] for ix in ixs[1:]]+[max_onset] #add end of last note
            for ix, span in enumerate(zip(start_onsets, stop_onsets)):
                weights[ix] = sum(beatstrengthgrid[span[0]:span[1]])
        else:
            weights = [beatstrength[ix] for ix in ixs]
        
        if syncopes:
            for ix, span in enumerate(zip(ixs, ixs[1:])):
                maxbeatstrength = np.max(beatstrengthgrid[onsettick[span[0]]:onsettick[span[1]]])
                weights[ix] = maxbeatstrength

        song['features']['weights'] = [0.0] * len(pitch40)
        for ix, songix in enumerate(ixs):
            song['features']['weights'][songix] = weights[ix]

        weightedpitch = np.zeros( (len(ixs), 40) )
        for ix, songix in enumerate(ixs):
            p = pitch40[songix]
            w = weights[ix]
            weightedpitch[ix, (p-1)%40] = w
        return weightedpitch, ixs

    def getBeatinsongFloat(self):
        """Convert `beatinsong` from Fraction to float

        Returns
        -------
        numpy vector
            Length is length of `ixs`. numpy vector with beatinsong as float
        """
        song = self.song.mtcsong
        beatinsong_float = np.zeros( len(self.ixs) )
        for ix, song_ix in enumerate(self.ixs):
            beatinsong_float[ix] = float(Fraction(song['features']['beatinsong'][song_ix]))
        return beatinsong_float

    def computePitchContext(self):   
        """Compute for each note a pitchcontext vector

        Returns
        -------
        numpy array
            Dimension is (length of `ixs`, 80). The first dimension corresponds to the note indices in
            `ixs`. The second dimension correpsonds to 40 pitches in the preceding context [:40] and
            40 pitches in the following context [40:]. Pitches are in base40 encoding.
        """
        #put param values in local variables for readibility
        len_context_beat = self.params['len_context_beat']
        use_metric_weights = self.params['use_metric_weights']
        use_distance_weights = self.params['use_distance_weights']
        min_distance_weight = self.params['min_distance_weight']
        includeFocus = self.params['includeFocus']
        partialNotes = self.params['partialNotes']
        normalize = self.params['normalize']
        epsilon = self.params['epsilon']
        
        song = self.song.mtcsong
        
        beatinsong = self.getBeatinsongFloat()
        songlength_beat = float(sum([Fraction(length) for length in song['features']['beatfraction']]))
        beatinsong_next = np.append(beatinsong[1:],songlength_beat)
        beatinsong_previous = np.insert(beatinsong[:-1],0, 0.0)

        if type(len_context_beat) == tuple or type(len_context_beat) == list:
            len_context_beat_pre = len_context_beat[0]
            len_context_beat_post = len_context_beat[1]
        elif len_context_beat == 'auto':
            print("Auto context length not yet implemented.")
        else:
            len_context_beat_pre = len_context_beat
            len_context_beat_post = len_context_beat
        len_context_beat = None

        if type(use_metric_weights) == tuple or type(use_metric_weights) == list:
            use_metric_weights_pre = use_metric_weights[0]
            use_metric_weights_post = use_metric_weights[1]
        else:
            use_metric_weights_pre = use_metric_weights
            use_metric_weights_post = use_metric_weights
        use_metric_weights = None

        if type(use_distance_weights) == tuple or type(use_distance_weights) == list:
            use_distance_weights_pre = use_distance_weights[0]
            use_distance_weights_post = use_distance_weights[1]
        else:
            use_distance_weights_pre = use_distance_weights
            use_distance_weights_post = use_distance_weights
        use_distance_weights = None

        if type(min_distance_weight) == tuple or type(min_distance_weight) == list:
            min_distance_weight_pre = min_distance_weight[0]
            min_distance_weight_post = min_distance_weight[1]
        else:
            min_distance_weight_pre = min_distance_weight
            min_distance_weight_post = min_distance_weight
        min_distance_weight = None

        #array to store the result
        pitchcontext = np.zeros( (len(self.ixs), 40 * 2) )

        contexts_pre = []
        contexts_post = []
        
        for ix, songix in enumerate(self.ixs):
            #compute offsets of all ohter notes
            beatoffset = beatinsong - beatinsong[ix]
            slicelength = beatinsong_next[ix] - beatinsong[ix]
            previous_slicelength = beatinsong[ix] - beatinsong_previous[ix]
            beatoffset_next = beatoffset - slicelength
            beatoffset_previous = beatoffset + previous_slicelength
            # print("noteix(song):", ixs[ix])
            # print()
            # print("beatinsong", beatinsong)
            # print("beatinsong_next", beatinsong_next)
            # print("beatoffset", beatoffset)
            # print("slicelength", slicelength)
            # print("beatoffset_next", beatoffset_next)
            #get context for each note
            #N.B. for some reason, np.where returns a tuple e.g: (array([], dtype=int64),)
            #for post, start context at END of focus note (anyway)
            if includeFocus in ['none', 'post']:
                context_pre = np.where(np.logical_and(beatoffset>=-(len_context_beat_pre + epsilon), beatoffset<0))[0]
            else: # ['both', 'pre']
                context_pre = np.where(np.logical_and(beatoffset>=-(len_context_beat_pre + epsilon), beatoffset<=0))[0]
            if includeFocus in ['none', 'pre']:
                #start context at END of note
                #do not include the note that starts AT the end of the context
                context_post = np.where(np.logical_and(beatoffset_next>=0, beatoffset_next<(len_context_beat_post - epsilon)))[0]
            else: # ['both', 'post']
                context_post = np.where(np.logical_and(beatoffset>=0, beatoffset_next<(len_context_beat_post - epsilon)))[0]        

            if partialNotes:
                if ix>0: #skip first, has no context_pre
                    #check wether context start at beginning of a note. If not, add previous note
                    #print(context_pre[0][0],beatoffset[context_pre[0][0]],len_context_beat)
                    if context_pre.shape[0]>0:
                        if np.abs( beatoffset[context_pre[0]] + len_context_beat_pre ) > epsilon:
                            if context_pre[0]-1 >= 0:
                                context_pre = np.insert(context_pre, 0, context_pre[0]-1)
                    else:
                        context_pre = np.insert(context_pre, 0, ix-1) #if context was empty, add previous note anyway

            contexts_pre.append(context_pre)
            contexts_post.append(context_post)

            #compute distance-weights
            if use_distance_weights_pre:
                distance_weights_pre  = beatoffset_previous[context_pre] * (1.0-min_distance_weight_pre)/len_context_beat_pre + 1.0
                #set negative weights to zero:
                distance_weights_pre[distance_weights_pre<0.0] = 0.0
            else:
                distance_weights_pre  = np.ones((context_pre.shape))

            if use_distance_weights_post:
                distance_weights_post = beatoffset_next[context_post] * -(1.0-min_distance_weight_post)/len_context_beat_post + 1.0
                #set negative weights to zero:
                distance_weights_post[distance_weights_post<0.0] = 0.0
            else:
                distance_weights_post = np.ones((context_post.shape))

            # print("ix", ix, ixs[ix])
            # print("length_context_pre", length_context_pre)
            # print("length_context_post", length_context_post)
            # print("distance_weights_pre", distance_weights_pre)
            # print("distance_weights_post", distance_weights_post)
            #combine context into one vector

            pitchcontext_pre  = np.dot(distance_weights_pre, self.weightedpitch[context_pre])
            pitchcontext_post = np.dot(distance_weights_post, self.weightedpitch[context_post])
            #normalize
            
            if not use_metric_weights_pre:
                pitchcontext_pre[pitchcontext_pre>0] = 1.0
            if not use_metric_weights_post:
                pitchcontext_post[pitchcontext_post>0] = 1.0
            
            if normalize:
                pitchcontext_pre /= np.sum(np.abs(pitchcontext_pre),axis=0)
                pitchcontext_post /= np.sum(np.abs(pitchcontext_post),axis=0)
            
            #store result
            pitchcontext[ix,:40] = pitchcontext_pre
            pitchcontext[ix,40:] = pitchcontext_post

        return pitchcontext, contexts_pre, contexts_post

    def printReport(
        self,
        note_ix=None, #report single note. IX in original song, not in ixs
        **features, #any other values to report. key: name, value: array size len(ixs)
    ):
        """Print for each note the values of several features to stdout.

        For each note print
        - pitch and (metric) weight as computed by `self.computeWeightedPitch`
        - indices (in `self.ixs`) of notes in the preceding context
        - indices (in the MTC features) of notes in the preceding context
        - indices (in `self.ixs`) of notes in the following context
        - indices (in the MTC features) of notes in the following context
        - pitches and corresponding weights in the precedings context
        - pitches and corresponding wieghts in the following context
        - any other feature provided as keyword argument (see below)

        Parameters
        ----------
        note_ix : int, default None
            Only print the values the note at index `note_ix` in the original melody (not in `self.ixs`).
        **features  : keyword arguments
            any other feature to report. The keyword is the name of the feature, the value is an 1D array
            with the same lenght as `self.ixs`.
        """
        for ix in range(len(self.ixs)):
            if note_ix:
                if note_ix != self.ixs[ix]: continue
            pre_pitches = []
            post_pitches = []
            for p in range(40):
                if self.pitchcontext[ix,p] > 0.0:
                    pre_pitches.append((base40[p],self.pitchcontext[ix,p]))
            for p in range(40):
                if self.pitchcontext[ix,p+40] > 0.0:
                    post_pitches.append((base40[p], self.pitchcontext[ix,p+40]))
            pre_pitches = [str(p) for p in sorted(pre_pitches, key=lambda x: x[1], reverse=True)]
            post_pitches = [str(p) for p in sorted(post_pitches, key=lambda x: x[1], reverse=True)]
            print("note", self.ixs[ix], "ix:", ix)
            print("  pitch:", self.song.mtcsong['features']['pitch'][self.ixs[ix]], self.song.mtcsong['features']['weights'][self.ixs[ix]])
            print("  context_pre (ixs):  ", self.contexts_pre[ix])
            print("  context_pre (notes):", np.array(self.ixs)[self.contexts_pre[ix]])
            print("  context_post (ixs):  ", self.contexts_post[ix])
            print("  context_post (notes):", np.array(self.ixs)[self.contexts_post[ix]])
            print("  pre:", "\n       ".join(pre_pitches))
            print("  post:", "\n        ".join(post_pitches))
            for name in features.keys():
                print(f"  {name}: {features[name][ix]}")
            print()

