import json
import tempfile

import numpy as np

import pitchcontext
from pitchcontext import Song, PitchContext
from pitchcontext.visualize import novelty2colordict, consonance2colordict, plotArray
from pitchcontext.models import computeConsonance, computeNovelty

path_to_krn = 'NLB147059_01.krn'
with open('NLB147059_01.json','r') as f:
    mtcsong = json.load(f)

song = Song(mtcsong, path_to_krn)

wpc = PitchContext(
    song,
    accumulateWeight=True,
    len_context_beat=(100.0,0.0),
    use_metric_weights=True,
    includeFocus='post',
    use_distance_weights=(False, False),
)
novelty = computeNovelty(song, wpc)
plotArray(novelty, wpc.ixs, 'note index', 'novelty');
cdict = novelty2colordict(novelty, wpc.ixs, 80, song.getSongLength())
with tempfile.TemporaryDirectory() as tmpdirname:
    song.showColoredPNG(cdict, tmpdirname, 'NLB147059_01_novelty', showfilename=False)
    song.showPNG()
