"""Long lived PyTorch A100 search worker; does not load or execute Inkling."""
from __future__ import annotations
import argparse, json, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from common import atomic_json, read_json, event, install_deadline

def quantize_grouped(w, bits, metadata_dtype='bfloat16'):
    # Numerical formulation follows MLX v0.32.4 CPU affine quantize.
    # Copyright Apple Inc.; original implementation and license are linked in README.
    low,high=w.amin(-1,keepdim=True),w.amax(-1,keepdim=True)
    mask=low.abs()>high.abs()
    scale=((high-low)/((1<<bits)-1)).clamp_min(1e-7)
    scale=torch.where(mask,scale,-scale)
    edge=torch.where(mask,low,high);q0=torch.round(edge/scale)
    scale=torch.where(q0!=0,edge/torch.where(q0!=0,q0,torch.ones_like(q0)),scale)
    bias=torch.where(q0!=0,edge,torch.zeros_like(edge))
    codes=torch.round((w-bias)/scale).clamp_(0,(1<<bits)-1)
    if metadata_dtype=='bfloat16':scale,bias=scale.to(torch.bfloat16).float(),bias.to(torch.bfloat16).float()
    reconstructed=codes*scale+bias
    return reconstructed,codes.to(torch.int64)

def clip(w, alpha, metadata_dtype='bfloat16'):
    if alpha==1:return w
    low,high=w.amin(-1,keepdim=True),w.amax(-1,keepdim=True)
    center,radius=(low+high)*.5,(high-low)*.5
    value=torch.minimum(torch.maximum(w,center-alpha*radius),center+alpha*radius)
    return value.to(torch.bfloat16).float() if metadata_dtype=='bfloat16' else value

def parity(root, device):
    metadata=read_json(root/'parity.json'); fixtures=np.load(root/'parity.npz',allow_pickle=False)
    worst=0.
    for record in metadata['records']:
        arr=torch.from_numpy(fixtures[record['weight']]).to(device)
        grouped=arr.reshape(arr.shape[0],-1,record['group'])
        dtype=record.get('metadata_dtype','bfloat16')
        output,codes=quantize_grouped(clip(grouped,metadata['alphas'][record['alpha_index']],dtype),record['bits'],dtype)
        observed=output.reshape(arr.shape).cpu().numpy()
        expected=fixtures[record['output']]
        if not np.allclose(observed,expected,atol=metadata['reconstruction_atol'],rtol=metadata['reconstruction_rtol']):
            raise ValueError(f'Reconstruction parity failed for {record}')
        flat=codes.reshape(arr.shape).cpu().numpy().astype(np.uint64)
        pack=32//record['bits']; flat=flat.reshape(arr.shape[0],-1,pack)
        packed=np.bitwise_or.reduce(flat << (np.arange(pack,dtype=np.uint64)*record['bits']),axis=-1).astype(np.uint32)
        if not np.array_equal(packed,fixtures[record['output']+'_packed']):
            raise ValueError(f'Packed code parity failed for {record}')
        worst=max(worst,float(np.max(np.abs(observed-expected))))
    return {'passed':True,'fixtures':len(metadata['records']),'max_abs_error':worst,
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        'torch':torch.__version__,'cuda':torch.version.cuda}

@torch.no_grad()
def search(w_numpy, stat_numpy, alphas, device, metadata_dtype='bfloat16'):
    source=torch.from_numpy(w_numpy).to(device=device,dtype=torch.float32)
    maps,metrics={},{}
    for group in (32,64,128):
        if source.shape[-1]%group:continue
        w=source.reshape(source.shape[0],-1,group)
        statistic=None if stat_numpy is None else torch.from_numpy(stat_numpy).to(device).reshape(1,-1,group)
        for bits in (2,4):
            best_c=torch.full(w.shape[:-1],float('inf'),device=device)
            best_w=best_c.clone();ic=torch.zeros_like(best_c,dtype=torch.uint8);iw=ic.clone()
            for ai,alpha in enumerate(alphas):
                reconstructed,_=quantize_grouped(clip(w,alpha,metadata_dtype),bits,metadata_dtype)
                square=(reconstructed-w).square()
                ec=square.sum(-1);ew=ec if statistic is None else (square*statistic).sum(-1)
                if ai==0:
                    metrics[f'R_g{group}_b{bits}']=float(ec.sum().item())
                    metrics[f'RW_g{group}_b{bits}']=float(ew.sum().item())
                ic=torch.where(ec<best_c,ai,ic);iw=torch.where(ew<best_w,ai,iw)
                best_c=torch.minimum(ec,best_c);best_w=torch.minimum(ew,best_w)
            for name,ids,loss in [('C',ic,best_c),('W',iw,best_w)]:
                key=f'{name}_g{group}_b{bits}';maps[key]=ids.cpu().numpy();metrics[key]=float(loss.sum().item())
    return maps,metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=10800);parser.add_argument('--cpu-test',action='store_true')
    parser.add_argument('--parity-only',action='store_true')
    args=parser.parse_args();install_deadline(args.seconds)
    root=args.run; worker=os.environ.get('SLURM_JOB_ID',str(os.getpid()))
    device=torch.device('cpu' if args.cpu_test else 'cuda:0')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        checked=parity(root,device);atomic_json(root/f'gpu-parity-{worker}.json',checked)
        event('gpu_parity_passed',**checked)
    except BaseException as error:
        atomic_json(root/f'gpu-parity-{worker}.json',{'passed':False,'error':repr(error),'traceback':traceback.format_exc()})
        raise
    if args.cpu_test or args.parity_only:return
    plan=read_json(root/'RUN_MANIFEST.json');alphas=plan['clipping']['alphas']
    while True:
        heartbeat=root/'heartbeat.json'
        if not heartbeat.exists() or time.time()-heartbeat.stat().st_mtime>180:
            event('controller_heartbeat_expired');return
        if (root/'STOP_WORKERS').exists():return
        ready=sorted((root/'queue').glob('*.ready.json'))
        if not ready:
            if (root/'PRODUCER_DONE').exists():return
            time.sleep(1);continue
        claim=None
        for item in ready:
            target=item.with_name(item.name.replace('.ready.json',f'.running-{worker}.json'))
            try:item.rename(target);claim=target;break
            except FileNotFoundError:continue
        if claim is None:continue
        task=read_json(claim); identity=task['id'];start=time.monotonic()
        try:
            with np.load(root/'queue'/task['payload'],allow_pickle=False) as payload:
                weights=payload['weights'];stat=payload['statistic'] if task['weighted'] else None
            maps,metrics=search(weights,stat,alphas,device,task.get('metadata_dtype','bfloat16'))
            output=root/'search'/(identity+'.npz');temporary=output.with_suffix('.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,**maps)
            temporary.replace(output)
            atomic_json(root/'search'/(identity+'.json'),{'status':'complete','task':task,'metrics':metrics,
                'worker':worker,'seconds':time.monotonic()-start})
            (root/'queue'/task['payload']).unlink(missing_ok=True);claim.unlink()
        except BaseException as error:
            atomic_json(root/'search'/(identity+'.error.json'),{'task':task,'error':repr(error),'traceback':traceback.format_exc()})
            claim.unlink(missing_ok=True)
            if isinstance(error,(KeyboardInterrupt,TimeoutError)):raise
        if torch.cuda.is_available():torch.cuda.empty_cache()

if __name__=='__main__':main()
