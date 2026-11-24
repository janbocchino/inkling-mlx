"""Exact MLX CPU clipping and parity fixtures for the CUDA search."""
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from common import tensor_id, atomic_json

ALPHAS = [1., .99, .98, .97, .95, .925, .90, .85, .80, .75, .70]

def apply_clipping(weights, indices, group, alphas=ALPHAS):
    original = weights.astype(mx.float32).reshape(weights.shape[0], -1, group)
    low, high = original.min(-1, keepdims=True), original.max(-1, keepdims=True)
    center, radius = (low + high) * .5, (high - low) * .5
    alpha = mx.array(alphas, dtype=mx.float32)[mx.array(indices.astype(np.int32))][..., None]
    clipped=mx.clip(original, center - alpha*radius, center + alpha*radius)
    clipped=mx.where(alpha==1,original,clipped)
    return clipped.reshape(weights.shape).astype(weights.dtype)

def cpu_search(weights, statistics=None, alphas=ALPHAS):
    maps, metrics = {}, {}
    w = weights; source=w.astype(mx.float32)
    for group in (32,64,128):
        if w.shape[-1] % group: continue
        source_grouped=source.reshape(w.shape[0], -1, group)
        weight = None
        if statistics is not None:
            weight=mx.array(statistics,dtype=mx.float32).reshape(1,-1,group)
        for bits in (2,4):
            best_c=mx.full(source_grouped.shape[:-1], float('inf'))
            best_w=best_c
            idx_c=mx.zeros(best_c.shape,dtype=mx.uint8);idx_w=idx_c
            first_c=first_w=None
            for ai in range(len(alphas)):
                ids=np.full(best_c.shape, ai, np.uint8)
                clipped=apply_clipping(w,ids,group,alphas)
                packed,s,b=mx.quantize(clipped,group_size=group,bits=bits)
                reconstructed=mx.dequantize(packed,s.astype(mx.float32),b.astype(mx.float32),group_size=group,bits=bits)
                square=mx.square(reconstructed-source).reshape(source_grouped.shape)
                ec=square.sum(-1); ew=ec if weight is None else (square*weight).sum(-1)
                if ai==0:first_c,first_w=ec.sum(),ew.sum()
                improve_c=ec<best_c;improve_w=ew<best_w
                idx_c=mx.where(improve_c,mx.array(ai,dtype=mx.uint8),idx_c)
                idx_w=mx.where(improve_w,mx.array(ai,dtype=mx.uint8),idx_w)
                best_c=mx.minimum(ec,best_c);best_w=mx.minimum(ew,best_w)
                mx.eval(idx_c,idx_w,best_c,best_w)
            for method,idx,loss in [('C',idx_c,best_c),('W',idx_w,best_w)]:
                key=f'{method}_g{group}_b{bits}'
                maps[key]=np.asarray(idx)
                metrics[key]=float(loss.sum().item())
            metrics[f'R_g{group}_b{bits}']=float(first_c.item())
            metrics[f'RW_g{group}_b{bits}']=float(first_w.item())
    return maps,metrics

def parity_fixtures(destination):
    rng=np.random.default_rng(20261121)
    arrays=[np.zeros((4,256),np.float32),np.full((4,256),1.5,np.float32),
        np.full((4,256),-.75,np.float32),rng.normal(0,.1,(16,256)).astype(np.float32)]
    arrays[-1][0,0]=8.;arrays[-1][1,10]=-11.
    records=[]; saved={}
    for variant in range(8):
        wi=variant%4;arr=arrays[wi];dtype=mx.bfloat16 if variant<4 else mx.float32
        name=f'w{variant}';w=mx.array(arr).astype(dtype);saved[name]=np.asarray(w.astype(mx.float32))
        for group in (32,64,128):
            for bits in (2,4):
                for ai in [0,4,8,10]:
                    idx=np.full((arr.shape[0],arr.shape[1]//group),ai,np.uint8)
                    clipped=apply_clipping(w,idx,group)
                    q,s,b=mx.quantize(clipped,group_size=group,bits=bits)
                    value=mx.dequantize(q,s.astype(mx.float32),b.astype(mx.float32),group_size=group,bits=bits)
                    mx.eval(value,q)
                    key=f'{name}_g{group}_b{bits}_a{ai}'
                    saved[key]=np.asarray(value)
                    saved[key+'_packed']=np.asarray(q)
                    records.append({'weight':name,'group':group,'bits':bits,'alpha_index':ai,'output':key,
                        'metadata_dtype':'bfloat16' if dtype==mx.bfloat16 else 'float32'})
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    np.savez(destination/'parity.npz',**saved)
    atomic_json(destination/'parity.json',{'records':records,'alphas':ALPHAS,
        'require_packed_codes_exact':True,'reconstruction_atol':2e-7,'reconstruction_rtol':1e-6})

def clipped_expert(weights, search_dir, path, expert, method, group, bits):
    directory=Path(search_dir)
    with np.load(directory/(tensor_id(path,expert)+'.npz'),allow_pickle=False) as maps:
        indices=maps[f'{method}_g{group}_b{bits}']
    return apply_clipping(weights,indices,group)
