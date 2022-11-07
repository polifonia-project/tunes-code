import music21 as m21
from MTCFeatures import MTCFeatureLoader

song = m21.converter.parse('/Users/krane108/data/MTC/MTC-FS-INST-2.0/krn/NLB150974_01.krn') #path to MTC

fl = MTCFeatureLoader('MTC-FS-INST-2.0')
seq_iter = fl.applyFilter(('inNLBIDs', ['NLB150974_01']))

for s in seq_iter:
    pitch = s['features']['pitch']
    imaw = s['features']['imaweight']

ix = 0
for n in song.flat.notes:
    if type(n.duration) != m21.duration.GraceDuration: #skip grace notes
        n.addLyric(imaw[ix])
        n.addLyric(pitch[ix])
        n.addLyric(n.beatStrength)
        ix = ix+1


song.show()