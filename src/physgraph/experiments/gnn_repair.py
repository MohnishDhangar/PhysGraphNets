"""Validation-only ablations for the diagnosed graph approximation bottlenecks."""
from pathlib import Path
from time import perf_counter
from physgraph.io.manifest import write_manifest
from physgraph.training.gnn import train
from physgraph.experiments.gnn_study import evaluate_checkpoint


def run_ablations(*,dataset,normalization,output,epochs=300):
    dest=Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('ablation output must be empty')
    dest.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    variants=[('local_mse','local','mse'),('local_h1','local','sobolev'),
              ('global_mse','global_square','mse'),('global_h1','global_square','sobolev')]
    config=dict(dataset=str(dataset),normalization=str(normalization),epochs=epochs,seed=0,
                width=32,depth=3,batch_size=7,learning_rate=.001,decay_every=100,decay_factor=.5,
                variants=variants,selection='validation n=8 only; no test samples evaluated')
    rows=[]
    write_manifest(dest/'manifest.json',config,{'status':'running','runs':rows},0.)
    for name,architecture,objective in variants:
        print('Variant',name,flush=True)
        train(dataset=dataset,normalization=normalization,output=dest/name,epochs=epochs,
              width=32,depth=3,batch_size=7,seed=0,learning_rate=.001,architecture=architecture,
              objective=objective,h1_weight=1.,decay_every=100,decay_factor=.5,
              progress=lambda row:print(row,flush=True) if row['epoch']%50==0 else None)
        metrics=evaluate_checkpoint(checkpoint=dest/name/'checkpoint.pt',dataset=dataset,
            normalization=normalization,output=dest/(name+'_validation'),criteria={},split='validation',resolutions=[8])
        rows.append({'variant':name,'validation':metrics})
        write_manifest(dest/'manifest.json',config,{'status':'running','runs':rows,'test_evaluated':False},perf_counter()-started)
    return write_manifest(dest/'manifest.json',config,{'status':'completed','runs':rows,'test_evaluated':False},perf_counter()-started)
