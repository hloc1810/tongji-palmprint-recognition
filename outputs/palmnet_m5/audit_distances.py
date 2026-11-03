"""Post-run distance audit; no model fitting or parameter selection."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from palmprint.training.palmnet import load_csr, nearest_class_scores
from palmprint.evaluation.metrics import ranking
from palmprint.utils import atomic_json

out=Path('outputs/palmnet_m5')
pred=pd.read_csv(out/'nearest_template/predictions.csv')
indices=np.unique(np.r_[pred.index[pred.correct_rank1 == 0].to_numpy(), [0, 1999, 5999]])
gallery=load_csr(out/'cache/gallery')
probe=load_csr(out/'cache/probe')[indices]
labels=np.load('artifacts/palmnet_m5/gallery_ids.npy',allow_pickle=False)
start=time.perf_counter()
classes,actual=nearest_class_scores(probe,gallery,labels,batch_size=2,gallery_batch_size=256)
with np.load(out/'nearest_template/scores.npz',allow_pickle=False) as saved:
    expected=saved['scores'][indices]
    np.testing.assert_array_equal(classes,saved['classes'])
np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=.1)
a,_=ranking(actual,classes); b,_=ranking(expected,classes)
np.testing.assert_array_equal(a[:,:5],b[:,:5])
result={'status':'passed','scope':'post-run verification only; no tuning',
        'sample_uids':pred.iloc[indices].sample_uid.tolist(),
        'max_absolute_distance_roundoff':float(np.max(np.abs(actual-expected))),
        'all_600_class_distances_compared':True,'top5_rankings_identical':True,
        'seconds':time.perf_counter()-start}
atomic_json(out/'distance_audit.json',result)
print(json.dumps(result,indent=2))
