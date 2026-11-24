"""Compare tiny complete CPU and hybrid CUDA forwards before full model use."""
import os,tempfile,json,sys
from pathlib import Path
import mlx.core as mx
from test_converter import make_source
from model_tools import load_reference,load_candidate,Statistics
from matrix_client import attach
from convert_inkling import convert
from common import atomic_json,event

def main():
    mx.set_default_device(mx.cpu)
    mx.random.seed(20261121)
    endpoint=os.environ.pop('SERIES_GPU_ENDPOINT')
    result={'status':'running','comparisons':[]}
    try:
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir();make_source(source)
            stats=Statistics();reference,_=load_reference(source,stats)
            for model_type in ['reference','affine_candidate']:
                if model_type=='reference':model=reference
                else:
                    os.environ.pop('SERIES_GPU_ENDPOINT',None)
                    convert(source,Path(d)/'candidate',64)
                    model,_=load_candidate(Path(d)/'candidate')
                inputs=[(mx.array([[1,2,3]*22]),{}),(mx.array([[1,127,2]]),{'pixel_values':[mx.ones((3,4,4))]})]
                outputs=[]
                for ids,media in inputs:
                    if model_type=='reference':stats.begin_case()
                    y=model(ids,**media);mx.eval(y);outputs.append(y)
                os.environ['SERIES_GPU_ENDPOINT']=endpoint
                attach(model,source_reference=model_type=='reference')
                for i,(ids,media) in enumerate(inputs):
                    if model_type=='reference':stats.begin_case()
                    y=model(ids,**media);mx.eval(y)
                    delta=float(mx.max(mx.abs(y-outputs[i])).item())
                    if not mx.allclose(y,outputs[i],atol=5e-5,rtol=5e-5).item():raise ValueError('Hybrid full forward mismatch')
                    result['comparisons'].append({'model':model_type,'image':i==1,'max_abs_error':delta})
                # Release the connection before resetting the service for the next model.
                model._matrix_rpc_client.connection.close()
        result['status']='passed';event('hybrid_tiny_parity_passed',**result)
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:
        os.environ['SERIES_GPU_ENDPOINT']=endpoint
        atomic_json(Path(endpoint).parent/'HYBRID_PARITY.json',result)

if __name__=='__main__':main()
