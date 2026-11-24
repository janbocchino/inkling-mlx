"""Use CUDA for selected matrix operators without porting the model."""
from __future__ import annotations
import json, os, socket, struct, time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.layers.moe import SwitchLinear,QuantizedSwitchLinear
from common import read_json,event

def exact(connection,n):
    result=bytearray(n);view=memoryview(result);position=0
    while position<n:
        got=connection.recv_into(view[position:],min(n-position,8*1024**2))
        if not got:raise EOFError('Matrix service connection closed')
        position+=got
    return result

class Client:
    def __init__(self,path):
        until=time.monotonic()+600
        while True:
            endpoint=read_json(path,{})
            try:
                if endpoint.get('status')!='ready':raise ConnectionError('CUDA matrix service is unavailable')
                self.token=endpoint['token'];self.connection=socket.create_connection((endpoint['host'],endpoint['port']),timeout=20)
                break
            except (OSError,ConnectionError):
                if time.monotonic()>until:raise
                time.sleep(3)
        self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        self.connection.settimeout(600);self.request({'operation':'reset'})
    def request(self,header,arrays=None):
        arrays=arrays or {};header={**header,'token':self.token}
        arrays={k:np.ascontiguousarray(v) for k,v in arrays.items()}
        header['arrays']=[{'name':k,'dtype':v.dtype.str,'shape':list(v.shape)} for k,v in arrays.items()]
        raw=json.dumps(header).encode();self.connection.sendall(struct.pack('!I',len(raw))+raw)
        for value in arrays.values():self.connection.sendall(memoryview(value).cast('B'))
        size=struct.unpack('!I',exact(self.connection,4))[0]
        result=json.loads(exact(self.connection,size));payload={}
        for entry in result.pop('arrays',[]):
            dtype=np.dtype(entry['dtype']);shape=tuple(entry['shape']);n=int(np.prod(shape))*dtype.itemsize
            payload[entry['name']]=np.frombuffer(exact(self.connection,n),dtype=dtype).reshape(shape)
        if not result.get('ok'):raise RuntimeError(result.get('error','CUDA matrix call failed'))
        return result,payload
    def compute(self,name,x,indices=None):
        mx.eval(x)
        arrays={'x':np.asarray(x.astype(mx.float32))}
        if indices is not None:
            mx.eval(indices);arrays['indices']=np.asarray(indices).astype(np.int32)
        _,payload=self.request({'operation':'compute','path':name},arrays)
        return mx.array(payload['result'])

class RPCLinear(nn.Linear):
    def __call__(self,x):return self._rpc.compute(self._path,x)
class RPCQuantizedLinear(nn.QuantizedLinear):
    def __call__(self,x):return self._rpc.compute(self._path,x)
class RPCSwitch(SwitchLinear):
    def __call__(self,x,indices,sorted_indices=False):
        collector=getattr(self,'_collector',None)
        if collector is not None:collector.observe(self._path,x,indices,self.num_experts)
        return self._rpc.compute(self._path,x,indices)
class RPCQuantizedSwitch(QuantizedSwitchLinear):
    def __call__(self,x,indices,sorted_indices=False):return self._rpc.compute(self._path,x,indices)

def attach(model,source_reference=False):
    endpoint=os.environ.get('SERIES_GPU_ENDPOINT')
    if not endpoint:return False
    client=Client(Path(endpoint));modules=list(model.named_modules());registered=0
    for path,module in modules:
        if isinstance(module,QuantizedSwitchLinear):kind,cls='qexpert',RPCQuantizedSwitch
        elif isinstance(module,SwitchLinear):kind,cls='expert',RPCSwitch
        elif isinstance(module,nn.QuantizedLinear):kind,cls='qlinear',RPCQuantizedLinear
        elif isinstance(module,nn.Linear):kind,cls='linear',RPCLinear
        else:continue
        header={'operation':'register','path':path,'kind':kind,
            'input_dims':module.input_dims if isinstance(module,(SwitchLinear,QuantizedSwitchLinear)) else module.weight.shape[-1]}
        arrays={}
        if kind.startswith('q'):
            header.update(bits=module.bits,group=module.group_size)
            header['input_dims']=module.scales.shape[-1]*module.group_size
            arrays['weight']=np.asarray(module.weight.view(mx.int32))
            arrays['scales']=np.asarray(module.scales.astype(mx.float32))
            arrays['biases']=np.asarray(module.biases.astype(mx.float32))
        else:
            # NVFP4 global scaling promotes reconstructed values to FP32. Keep
            # these exact values; the server caches full precision expert tiles.
            bf16=module.weight.dtype==mx.bfloat16
            header['storage']='bf16bits' if bf16 else 'float32'
            if bf16:arrays['weight']=np.asarray(module.weight.astype(mx.bfloat16).view(mx.int16))
            else:arrays['weight']=np.asarray(module.weight.astype(mx.float32))
        if 'bias' in module:arrays['bias']=np.asarray(module.bias.astype(mx.float32))
        result,_=client.request(header,arrays)
        module.__class__=cls;module._rpc=client;module._path=path;registered+=1
        if registered%32==0:event('cuda_matrices_registered',count=registered,allocated_bytes=result['allocated_bytes'])
    model._matrix_rpc_client=client
    event('cuda_matrix_backend_attached',modules=registered)
    return True
