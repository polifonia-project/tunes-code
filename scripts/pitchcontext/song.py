import copy
from fractions import Fraction
from math import gcd
import subprocess
import tempfile
import os

import numpy as np
import music21 as m21
m21.humdrum.spineParser.flavors['JRP'] = True
from IPython import display

#from datetime import datetime
#print(__file__, datetime.now().strftime("%H:%M:%S"))

#Exception: computed onsets do not match onsets in MTC
class OnsetMismatchError(Exception):
    def __init__(self, arg):
        self.arg = arg
    def __str__(self):
        return repr(self.arg)

#Exception: parsing failed
class ParseError(Exception):
    def __init__(self, arg):
        self.args = arg
    def __str__(self):
        return repr(self.value)

def lcm(a, b):
    """Computes the lowest common multiple.
    
    Parameters
    ----------
    a : int
    b : int

    Returns
    ------
    int
        Lowest common multiple of a and b.
    """
    return a * b // gcd(a, b)

def fraction_gcd(x, y):
    """Computes the greatest common divisor as Fraction
    
    Parameters
    ----------
    x : Fraction
    y : Fraction

    Returns
    ------
    Fraction
        greatest common divisor of x and y as Fraction.
    """
    a = x.numerator
    b = x.denominator
    c = y.numerator
    d = y.denominator
    return Fraction(gcd(a, c), lcm(b, d))

class Song:
    """Class for containing data about a song. An object of the class `Song` holds
    feature values for each note as provided by MTCFeatures, additionally computed
    features for computing the weighted pitch vectors, a music21 object representing
    the score.
    
    Parameters
    ----------
    mtcsong : dict
        Dictionary with feature values of all notes of the song, as provided by
        MTCFeatures
    krnfilename : string
        Filename of a corresponding **kern file

    Attributes
    ----------
    mtcsong : dict
        Dictionary with feature values of all notes of the song, as provided by
        MTCFeatures
    krnfilename : string
        Filename of a corresponding **kern file
    s : music21 Stream
        music21 object representing the score of the song. The following operation have
        been performed: ties removal, padding of incomplete bars, grace note removal.
        The  indices of the notes in the resulting stream should corresponsd to the indices
        of the feature values in `self.mtcsong`
    onsets : list of ints
        list with onset times. onsets[n] is the onset of the (n+1)th note. The term onset
        refers to the time in the score at which the note starts (in music21 this is offset).
        The resolution is determined by the number of ticks per quarter note
        as computed by getResolution().
    beatstrength_grid : list
        Contains a beatstrength for each possible onset as computed by the music21 meter model.
        Onsets are indices in the list. beatstrength_grid[n] is the beatstrength for a note
        starting at n.
    """
    #mtcson : dict from MTCFeatures
    #krnfilename : filename of the corresponding **kern file
    def __init__(self, mtcsong, krnfilename):
        """Instantiate an object of class Song.

        Parameters
        ----------
        mtcsong : dict
            Dictionary of feature values and metadata of a song as provided by MTCFeatures
        krnfilename : str
            Full file name of a corresponding **kern file.
        """
        self.mtcsong = copy.deepcopy(mtcsong)
        self.krnfilename = krnfilename
        self.s = self.parseMelody()
        self.onsets = self.getOnsets()
        self.beatstrength_grid = self.create_beatstrength_grid()
        self.add_features()

    def getSongLength(self):
        """Returns the number of notes in the song

        Returns
        -------
        int
            Number of notes in the song
        """
        return len(self.mtcsong['features']['pitch'])

    def getDurationUnit(self):
        """Returns a unit of note duration that is the greatest common divisor of all note durations.

        Returns
        -------
        Fraction
            Duration unit
        """
        sf = self.s.flat.notesAndRests
        unit = Fraction(sf[0].duration.quarterLength)
        for n in sf:
            unit = fraction_gcd(unit, Fraction(n.duration.quarterLength))
        return fraction_gcd(unit, Fraction(1,1)) # make sure 1 is dividable by the unit.denominator

    #return number of ticks per quarter note
    def getResolution(self) -> int:
        """Return the number of ticks per quarter note given the duration unit.

        Returns
        -------
        int
            number of ticks per quarter note.
        """
        unit = self.getDurationUnit()
        #number of ticks is 1 / unit (if that is an integer)
        ticksPerQuarter = unit.denominator / unit.numerator
        if ticksPerQuarter.is_integer():
            return int(unit.denominator / unit.numerator)
        else:
            print(self.s.filePath, ' non integer number of ticks per Quarter')
            return 0

    def getOnsets(self):
        """Returns a list of onsets (ints). Onsets are multiples of the duration unit.

        Returns
        -------
        list of int
            Onset for each note.

        Raises
        ------
        OnsetMismatchError
            Raised if the computed onsets do not match with the onsets as provided in MTCFeatures.
            These should be the same.
        """
        ticksPerQuarter = self.getResolution()
        onsets = [int(n.offset * ticksPerQuarter) for n in self.s.flat.notes]
        #check whether same onsets in songfeatures
        assert len(self.mtcsong['features']['onsettick']) == len(onsets)
        #NB. If initial rests (e.g. NLB142326_01) all onsets are shifted wrt MTC
        for ix in range(len(onsets)):
            if self.mtcsong['features']['onsettick'][ix] != onsets[ix] - onsets[0]:
                raise OnsetMismatchError("Onsets do not match. Probably due to a known bug in MTCFeatures.")
        return onsets

    # s : music21 stream
    def removeGrace(self, s):
        """Remove all grace notes from the music21 stream.

        Parameters
        ----------
        s : music21 Stream
            Object representing the score of the song.
        """
        #highest level:
        graceNotes = [n for n in s.recurse().notes if n.duration.isGrace]
        for grace in graceNotes:
            grace.activeSite.remove(grace)
        #if s is not flat, there will be Parts and Measures:
        for p in s.getElementsByClass(m21.stream.Part):
            #Also check for notes at Part level.
            #NLB192154_01 has grace note in Part instead of in a Measure. Might be more.
            graceNotes = [n for n in p.recurse().notes if n.duration.isGrace]
            for grace in graceNotes:
                grace.activeSite.remove(grace)
            for ms in p.getElementsByClass(m21.stream.Measure):
                graceNotes = [n for n in ms.recurse().notes if n.duration.isGrace]
                for grace in graceNotes:
                    grace.activeSite.remove(grace)
        
    # add left padding to partial measure after repeat bar
    def padSplittedBars(self, s):
        """Add padding to bars that originate from splitting a bar at a repeat sign. The second of the two
        resulting (partial) bars should have a padding equal to the lenght of the first (partial) bar in 
        order to obtain correct beatstrength values.

        Parameters
        ----------
        s : music21 Stream
            Object representing the score of the song.

        Returns
        -------
        music21 Stream
            s with padded bars
        """
        partIds = [part.id for part in s.parts] 
        for partId in partIds: 
            measures = list(s.parts[partId].getElementsByClass('Measure')) 
            for m in zip(measures,measures[1:]): 
                if m[0].quarterLength + m[0].paddingLeft + m[1].quarterLength == m[0].barDuration.quarterLength: 
                    m[1].paddingLeft = m[0].quarterLength 
        return s

    #N.B. contrary to the function currently in MTCFeatures (nov 2022), do not flatten the stream
    def parseMelody(self):
        """Converts **kern to music21 Stream and do necessary preprocessing:
        - pad splitted bars
        - strip ties
        - remove grace notes
        The notes in the resulting score correspond 1-to-1 with the notes in MTCFeatures.

        Returns
        -------
        music21 Stream
            music21 score for the song.

        Raises
        ------
        ParseError
            Raised if the **kern file is unparsable.
        """
        try:
            s = m21.converter.parse(self.krnfilename)
        except m21.converter.ConverterException:
            raise ParseError(self.krnfilename)
        #add padding to partial measure caused by repeat bar in middle of measure
        s = self.padSplittedBars(s)
        s = s.stripTies()
        self.removeGrace(s)
        return s

    #Add metric grid
    def create_beatstrength_grid(self):
        """Creates a vector with for each possible onset the beatstrength. The last onset corresponds to the
        end of the last note.

        Returns
        -------
        list (float)
            list with beatstrengths. The beatstrenght for onset n is beatstrength_grid[n].
        """
        beatstrength_grid = []
        unit = Fraction(1, self.getResolution()) #duration of a tick
        s_onsets = m21.converter.parse(self.krnfilename) #original score
        for p in s_onsets.getElementsByClass(m21.stream.Part):
            for m in p.getElementsByClass(m21.stream.Measure):
                offset = 0*unit
                while offset < m.quarterLength:
                    n = m21.note.Note("C").getGrace()
                    m.insert(offset, n)
                    beatstrength_grid.append(n.beatStrength)
                    offset += unit
        assert beatstrength_grid[self.onsets[-1]] == self.mtcsong['features']['beatstrength'][-1]
        return beatstrength_grid

    def add_features(self):
        """Adds a few features that are needed for computing pitch vectors. One value for each note.
        - syncope: True if the note is a syncope (there is a a higher metric weight in the span of the note than at the start of the note).
        - maxbeatstrength: the highest beatstrenght DURING the note.
        """
        self.mtcsong['features']['syncope'] = [False] * len(self.mtcsong['features']['pitch'])
        self.mtcsong['features']['maxbeatstrength'] = [0.0] * len(self.mtcsong['features']['pitch'])
        self.mtcsong['features']['beatstrengthgrid'] = self.beatstrength_grid
        beatstrength_grid_np = np.array(self.beatstrength_grid)
        for ix, span in enumerate(zip(self.mtcsong['features']['onsettick'],self.mtcsong['features']['onsettick'][1:])):
            self.mtcsong['features']['maxbeatstrength'][ix] = self.mtcsong['features']['beatstrength'][ix]
            if np.max(beatstrength_grid_np[span[0]:span[1]]) > self.mtcsong['features']['beatstrength'][ix]:
                self.mtcsong['features']['syncope'][ix] = True
                self.mtcsong['features']['maxbeatstrength'][ix] = np.max(beatstrength_grid_np[span[0]:span[1]])
        #final note:
        self.mtcsong['features']['maxbeatstrength'][-1] = self.mtcsong['features']['beatstrength'][-1]
    
    def getColoredSong(self, colordict):
        """Create a new music21 stream with notes colored according to `colordict`.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.

        Returns
        -------
        music21 Stream
            music21 Stream.
        """
        s = self.parseMelody()
        #check for right length #if so, assume notes correspond with features
        assert self.getSongLength() == len(s.flat.notes)
        for color, ixs in colordict.items():
            for ix in ixs:
                s.flat.notes[int(ix)].style.color = color
        #add index of note as lyric
        for ix, n in enumerate(s.flat.notes):
            n.lyric = None
            n.addLyric(str(ix))
        return s
    
    #we need to repair lily generated by m21 concerning color
    #\override Stem.color -> \once\override Stem.color
    #\override NoteHead.color -> \once\override NoteHead.color

    def repairLyline(self, line):
        """Corrects possbile error in a line of the Ly export. Note coloring should be done once.

        Parameters
        ----------
        line : str
            a line of a generated lilypond file.

        Returns
        -------
        str
            corrected line
        """
        line = line.replace("\\override Stem.color","\\once\\override Stem.color")
        line = line.replace("\\override NoteHead.color","\\once\\override NoteHead.color")
        line = line.replace("\\include \"lilypond-book-preamble.ly\"","")
        return line
    
    def repairLy(self, filename):
        """Go over a lilypond file, and correct the lines (see `self.repairLyline`).

        Parameters
        ----------
        filename : str
            Filename of the lilypond file to repair.
        """
        with open(filename,'r') as f:
            lines = [self.repairLyline(l) for l in f.readlines()]
        lines = lines + [ f'\paper {{ tagline = "" }}']
        with open(filename,'w') as f:
            f.writelines(lines)

    def insertFilenameLy(self, filename):
        """Insert a header with the filename into a lilypond source file.

        Parameters
        ----------
        filename : str
            Filename of the lilypond file to process.
        """
        with open(filename,'r') as f:
            lines = [l for l in f.readlines()]
        with open(filename,'w') as f:
            for l in lines:
                if "\\score" in l:
                    f.write(f'\\header {{ opus = "{filename}" }}\n\n')
                f.write(l)

    def createColoredPDF(self, colordict, outputpath, filebasename=None, showfilename=True):
        """Create a pdf with a score with colored notes.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.
        outputpath : str
            name of the output directory
        filebasename : str, default None
            basename of the pdf file to generate (without .pdf). If None, the identifier of the song as provided by
            MTCFeatures is used as file name.
        showfilename : bool, default True
            Include the filename in the pdf (lilypond opus header).

        Returns
        -------
        path-like object
            Full path of the generated pdf.
        """
        if filebasename is None:
            filebasename = self.mtcsong['id']
        s = self.getColoredSong(colordict)
        s.write('lily', os.path.join(outputpath, filebasename+'.ly'))
        self.repairLy(os.path.join(outputpath, filebasename+'.ly'))
        if showfilename:
            self.insertFilenameLy(os.path.join(outputpath, filebasename+'.ly'))
        output = subprocess.run(["lilypond", os.path.join(outputpath, filebasename+'.ly')], cwd=outputpath, capture_output=True)
        return os.path.join(outputpath, filebasename+'.pdf')

    def createColoredPNG(self, colordict, outputpath, filebasename=None, showfilename=True):
        """Create a png with a score with colored notes.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.
        outputpath : str
            name of the output directory
        filebasename : str, default None
            basename of the png file to generate (without .png). If None, the identifier of the song as provided by
            MTCFeatures is used as file name.
        showfilename : bool, default True
            Include the filename in the png (lilypond opus header).

        Returns
        -------
        path-like object
            Full path of the generated png.
        """
        pdf_fn = self.createColoredPDF(colordict, outputpath, filebasename, showfilename)
        png_fn = pdf_fn.replace('.pdf','.png')
        output = subprocess.run(['convert', '-density', '100', pdf_fn, '-alpha', 'Remove', '-trim', png_fn], cwd=outputpath, capture_output=True)
        return png_fn
    
    def showColoredPNG(self, colordict, outputpath, filebasename=None, showfilename=True):
        """Show a png with a score with colored notes. For use in a Jupyter notebook.

        Parameters
        ----------
        colordict : dict
            The keys are the colors, the values the indices of the notes with that color. E.g. {'red':[0,10,11],'grey':[-1]}
            colors notes at indices 0, 10, and 11 red, and the last note grey.
        outputpath : str
            name of the output directory
        filebasename : str, default None
            basename of the png file to generate (without .png). If None, the identifier of the song as provided by
            MTCFeatures is used as file name.
        showfilename : bool, default True
            Include the filename in the png (lilypond opus header).
        """
        png_fn = self.createColoredPNG(colordict, outputpath, filebasename, showfilename)
        display.display(display.Image(png_fn))

    def showPNG(self):
        """Show a png with a score of the song. For use in a Jupyter notebook.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.showColoredPNG({}, tmpdirname, showfilename=False)
    
    def createPNG(self, outputpath, filebasename=None, showfilename=False):
        """Creates a PNG image of the score.

        Parameters
        ----------
        outputpath : str
            name of the output directory
        filebasename : str, default None
            basename of the png file to generate (without .png). If None, the identifier of the song as provided by
            MTCFeatures is used as file name.
        showfilename : bool, default True
            Include the filename in the png (lilypond opus header).

        Returns
        -------
        path-like object
            Full path of the generated png.
        """
        return self.createColoredPNG({}, outputpath, filebasename=filebasename, showfilename=showfilename)
