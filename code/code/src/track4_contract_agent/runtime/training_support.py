"""Unlabelled image-vector support. This is not a calibrated OOD classifier."""
from pathlib import Path
import hashlib,json
import numpy as np


class TrainingSupport:
    def __init__(self,path):
        self.path=Path(path)
        with np.load(self.path,allow_pickle=False) as z:
            self.vectors=np.asarray(z['vectors'],dtype=np.float32)
            self.categories=[str(c) for c in z['categories']]
        self.meta=json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'))
        if self.vectors.ndim!=2 or self.vectors.shape[1]!=2048 or len(self.vectors)!=len(self.categories):
            raise ValueError('Invalid training support shape')
        if not np.isfinite(self.vectors).all():raise ValueError('Non-finite training support')
        norm=np.linalg.norm(self.vectors,axis=1,keepdims=True)
        if np.any(norm<1e-8):raise ValueError('Zero support vector')
        self.vectors=self.vectors/norm
        if set(self.categories)!={'桥梁','轨道'}:raise ValueError('Incomplete support categories')
        if self.meta['source_scope']!='official_training_images_only':raise ValueError('Unsafe reference scope')

    def annotate(self,vectors,evidence):
        x=np.asarray(vectors,dtype=np.float32)
        if x.shape!=(len(evidence),2048) or not np.isfinite(x).all():raise ValueError('Support input alignment')
        x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
        for category in ('桥梁','轨道'):
            query=[i for i,e in enumerate(evidence) if e['question_category']==category]
            refs=[i for i,c in enumerate(self.categories) if c==category]
            if not query:continue
            nearest=(x[query]@self.vectors[refs].T).max(axis=1)
            threshold=float(self.meta['thresholds'][category])
            if not 0<threshold<1:raise ValueError('Invalid support threshold')
            for i,similarity in zip(query,nearest):
                evidence[i].update(training_support_similarity=float(similarity),
                                   training_support_threshold=threshold,
                                   low_training_support=bool(similarity<threshold))
        return {'scope':self.meta['source_scope'],'n_references':len(self.vectors),
                'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'thresholds':self.meta['thresholds'],
                'n_low_support':sum(e.get('low_training_support',False) for e in evidence),
                'calibrated_ood_probability':False}
