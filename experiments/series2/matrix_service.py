"""Authenticated intra-cluster CUDA matrix backend for the CPU MLX model.

Only numeric arrays and JSON pass over the connection. Model control flow,
attention, image preprocessing, calibration and scoring remain in pinned MLX.
"""
from __future__ import annotations
import argparse, collections, json, os, secrets, socket, struct, time, traceback
from pathlib import Path
import numpy as np
import torch
from common import atomic_json,event,read_json

def exact(connection,n):
    result=bytearray(n);view=memoryview(result);position=0
    while position<n:
        got=connection.recv_into(view[position:],min(n-position,8*1024**2))
        if not got:raise EOFError('Connection closed')
        position+=got
    return result

def receive(connection):
    size=struct.unpack('!I',exact(connection,4))[0]
    if size>1024**2:raise ValueError('Oversized message header')
    header=json.loads(exact(connection,size));arrays={}
    for item in header.pop('arrays',[]):
        dtype=np.dtype(item['dtype']);shape=tuple(item['shape']);n=int(np.prod(shape))*dtype.itemsize
        if n>4*1024**3:raise ValueError('Oversized numeric payload')
        arrays[item['name']]=np.frombuffer(exact(connection,n),dtype=dtype).reshape(shape)
    return header,arrays

def send(connection,header,arrays=None):
    arrays=arrays or {};header=dict(header)
    contiguous={k:np.ascontiguousarray(v) for k,v in arrays.items()}
    header['arrays']=[{'name':k,'dtype':v.dtype.str,'shape':list(v.shape)} for k,v in contiguous.items()]
    raw=json.dumps(header).encode();connection.sendall(struct.pack('!I',len(raw))+raw)
    for value in contiguous.values():connection.sendall(memoryview(value).cast('B'))

class MatrixBackend:
    def __init__(self):
        self.modules={};self.calls=0;self.cache=collections.OrderedDict();self.cache_bytes=0
        self.cache_limit=48*1024**3
    def clear(self):
        self.modules.clear();self.cache.clear();self.cache_bytes=0;torch.cuda.empty_cache()
    def register(self,header,arrays):
        data={}
        for name,array in arrays.items():
            tensor=torch.from_numpy(array)
            if name=='weight' and header.get('storage')=='bf16bits':tensor=tensor.view(torch.bfloat16)
            # Exact FP32 source experts live in host RAM and enter a bounded CUDA
            # cache on demand. The 12 repeatedly visited core layers fit together.
            data[name]=tensor if name=='weight' and header['kind']=='expert' else tensor.to('cuda')
        self.modules[header['path']]={'data':data,'kind':header['kind'],'bits':header.get('bits'),
            'group':header.get('group'),'input_dims':header['input_dims'],'path':header['path']}
        return {'registered':header['path'],'allocated_bytes':torch.cuda.memory_allocated()}
    def matrix(self,module,expert=None):
        data=module['data'];w=data['weight'] if expert is None else data['weight'][expert]
        if module['kind']=='expert':
            key=(module['path'],expert)
            if key in self.cache:
                self.cache.move_to_end(key);return self.cache[key]
            needed=w.numel()*w.element_size()
            while self.cache_bytes+needed>self.cache_limit and self.cache:
                _,old=self.cache.popitem(last=False);self.cache_bytes-=old.numel()*old.element_size();del old
            value=w.to('cuda');self.cache[key]=value;self.cache_bytes+=needed
            return value
        if module['kind'] in ['qlinear','qexpert']:
            bits=module['bits'];factor=32//bits
            shifts=torch.arange(factor,device='cuda',dtype=torch.int32)*bits
            values=((w[...,None]>>shifts)&((1<<bits)-1)).reshape(w.shape[:-1]+(-1,)).float()
            s=data['scales'] if expert is None else data['scales'][expert]
            b=data['biases'] if expert is None else data['biases'][expert]
            values=values.reshape(values.shape[:-1]+(-1,module['group']))
            return (values*s.float()[...,None]+b.float()[...,None]).reshape(w.shape[:-1]+(module['input_dims'],))
        return w.float()
    @torch.no_grad()
    def compute(self,header,arrays):
        module=self.modules[header['path']];data=module['data'];x=torch.from_numpy(arrays['x']).to('cuda')
        if module['kind'] in ['linear','qlinear']:
            result=x@self.matrix(module).T
            if 'bias' in data:result=result+data['bias'].float()
        else:
            indices=arrays['indices'];m=x.shape[-2]
            target=indices.shape+(m,x.shape[-1]);x=x.expand(target).reshape(-1,x.shape[-1])
            flat=np.repeat(indices.reshape(-1),m)
            output_dims=data['weight'].shape[-2]
            result=torch.empty((len(flat),output_dims),dtype=torch.float32,device='cuda')
            for expert in np.unique(flat):
                positions=torch.from_numpy(np.flatnonzero(flat==expert)).to('cuda')
                value=x.index_select(0,positions)@self.matrix(module,int(expert)).T
                if 'bias' in data:value=value+data['bias'][int(expert)].float()
                result.index_copy_(0,positions,value)
            result=result.reshape(indices.shape+(m,output_dims))
        self.calls+=1
        return result.cpu().numpy()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seconds',type=int,default=42800)
    args=p.parse_args();os.umask(0o077)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    backend=MatrixBackend();token=secrets.token_hex(32)
    listener=socket.socket();listener.bind(('',0));listener.listen(1);listener.settimeout(10)
    endpoint={'host':socket.gethostname(),'port':listener.getsockname()[1],'token':token,
        'job_id':os.environ.get('SLURM_JOB_ID'),'device':torch.cuda.get_device_name(0),'status':'ready'}
    atomic_json(args.run/'GPU_ENDPOINT.json',endpoint)
    event('matrix_service_ready',host=endpoint['host'],port=endpoint['port'],gpu=endpoint['device'])
    started=time.time()
    try:
        while time.time()-started<args.seconds:
            if (args.run/'STOP_MATRIX_SERVICE').exists():break
            status=read_json(args.run/'STATUS.json',{})
            if status and status.get('status','').startswith('complete'):break
            heartbeat=args.run/'heartbeat.json'
            if status.get('status')=='running' and heartbeat.exists() and time.time()-heartbeat.stat().st_mtime>300:break
            try:connection,_=listener.accept()
            except socket.timeout:continue
            connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            connection.settimeout(600)
            try:
                while time.time()-started<args.seconds:
                    header,arrays=receive(connection)
                    if not secrets.compare_digest(header.get('token',''),token):raise ValueError('Bad matrix service authentication')
                    operation=header['operation']
                    if operation=='reset':
                        backend.clear();send(connection,{'ok':True})
                    elif operation=='register':send(connection,{'ok':True,**backend.register(header,arrays)})
                    elif operation=='compute':send(connection,{'ok':True},{'result':backend.compute(header,arrays)})
                    elif operation=='ping':send(connection,{'ok':True,'calls':backend.calls})
                    else:raise ValueError('Unknown matrix operation')
            except EOFError:pass
            except BaseException as error:
                try:send(connection,{'ok':False,'error':repr(error)})
                except Exception:pass
                event('matrix_connection_error',error=repr(error))
            finally:
                connection.close();backend.clear()
    finally:
        endpoint.update(status='stopped',calls=backend.calls);atomic_json(args.run/'GPU_ENDPOINT.json',endpoint)
        listener.close()

if __name__=='__main__':main()
