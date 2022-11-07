import music21 as m21
m21.humdrum.spineParser.flavors['JRP'] = True
from fractions import Fraction
from math import gcd

def lcm(a, b):
    """Return lowest common multiple."""
    return a * b // gcd(a, b)

def fraction_gcd(x, y):
    a = x.numerator
    b = x.denominator
    c = y.numerator
    d = y.denominator
    return Fraction(gcd(a, c), lcm(b, d))

def getDurationUnit(s):
    sf = s.flat.notesAndRests
    unit = Fraction(sf[0].duration.quarterLength)
    for n in sf:
        unit = fraction_gcd(unit, Fraction(n.duration.quarterLength))
    return fraction_gcd(unit, Fraction(1,1)) # make sure 1 is dividable by the unit.denominator

#return number of ticks per quarter note
def getResolution(s) -> int:
    unit = getDurationUnit(s)
    #number of ticks is 1 / unit (if that is an integer)
    ticksPerQuarter = unit.denominator / unit.numerator
    if ticksPerQuarter.is_integer():
        return int(unit.denominator / unit.numerator)
    else:
        print(s.filePath, ' non integer number of ticks per Quarter')
        return 0

def getOnsets(s):
    ticksPerQuarter = getResolution(s)
    onsets = [int(n.offset * ticksPerQuarter) for n in s.flat.notes]
    return onsets

# s : music21 stream
def removeGrace(s, flat=False):
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
def padSplittedBars(s):
    partIds = [part.id for part in s.parts] 
    for partId in partIds: 
        measures = list(s.parts[partId].getElementsByClass('Measure')) 
        for m in zip(measures,measures[1:]): 
            if m[0].quarterLength + m[0].paddingLeft + m[1].quarterLength == m[0].barDuration.quarterLength: 
                m[1].paddingLeft = m[0].quarterLength 
    return s

#N.B. contrary to the function currently in MTCFeatures (nov 2022), do not flatten the stream
def parseMelody(path):
    try:
        s = m21.converter.parse(path)
    except m21.converter.ConverterException:
        raise ParseError(path)
    #add padding to partial measure caused by repeat bar in middle of measure
    s = padSplittedBars(s)
    s = s.stripTies()
    removeGrace(s)
    return s