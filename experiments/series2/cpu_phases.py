"""Bounded MLX CPU phases launched by the persistent Slurm controller."""
from __future__ import annotations
import argparse, collections, gc, json, os, shutil, time, traceback
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model, load_tokenizer
from common import atomic_json, read_json, digest, sha_file, tensor_id, event, install_deadline
from model_tools import (Statistics, expert_modules, layer_number, load_reference, load_candidate,
    case_arrays, score_case, aggregate)
from convert_inkling import check_runtime, reconstruct_expert, MATRIX_CLASSES, choose_bits, matrix_dims
from clipping import ALPHAS, cpu_search

KIT=Path(__file__).resolve().parent

def pilot(args):
    import resource
    started=time.monotonic();report={'status':'running','source':str(args.source),'cases':[]}
    atomic_json(args.run/'PILOT.json',report)
    try:
        stats=Statistics();model,config=load_reference(args.source,stats)
        report['load_seconds']=time.monotonic()-started
        tokenizer=load_tokenizer(Path(args.source))
        for case in read_json(KIT/'data'/'calibration.json')[:3]:
            stats.begin_case();t=time.monotonic()
            inputs,media,_,_=case_arrays(case,KIT/'data',tokenizer,config,with_answer=False)
            logits=model(inputs,**media);mx.eval(logits)
            if not mx.all(mx.isfinite(logits)).item():raise ValueError('Nonfinite full source pilot logits')
            report['cases'].append({'id':case['id'],'modality':case['modality'],
                'tokens':inputs.shape[1],'seconds':time.monotonic()-t,'finite':True})
            atomic_json(args.run/'PILOT.json',report);event('full_source_pilot_case',**report['cases'][-1])
            del logits;gc.collect();mx.clear_cache()
        stats.enabled=False
        report['answer_loss_pilot']=score_case(model,config,tokenizer,read_json(KIT/'data'/'development.json')[0],KIT/'data')
        report['status']='passed'
        report['observed_expert_modules']=len(stats.counts)
        report['peak_rss_bytes_linux']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['seconds']=time.monotonic()-started;atomic_json(args.run/'PILOT.json',report)

def reference(args):
    root=args.run;data=KIT/'data';started=time.monotonic()
    stats=Statistics();accepted=[];records=[];errors=[]
    report={'status':'running','cpu_arithmetic':'FP32 MLX with authenticated CUDA matrix backend when configured','source_values':'NVFP4/MXFP8 reconstructed through BF16',
        'data_manifest_hash':sha_file(data/'DATA_MANIFEST.json')}
    atomic_json(root/'reference_metrics.json',report)
    try:
        model,config=load_reference(args.source,stats)
        tokenizer=load_tokenizer(Path(args.source))
        load_seconds=time.monotonic()-started
        event('reference_loaded',seconds=load_seconds)
        calibration=read_json(data/'calibration.json')
        token_count=0;calibration_start=time.monotonic();durations=[]
        for case in calibration:
            if time.monotonic()-started>min(7200,args.seconds-1800):break
            if token_count>=24576:break
            stats.begin_case();case_start=time.monotonic()
            try:
                inputs,media,_,_=case_arrays(case,data,tokenizer,config,with_answer=False)
                if inputs.shape[1]>384:
                    if media:raise ValueError('Image calibration example too long')
                    inputs=inputs[:,:384]
                embeddings=model.embed_inputs(inputs,**media) if media else None
                hidden=model.model.language_model(inputs,input_embeddings=embeddings)
                mx.eval(hidden)
                if not mx.all(mx.isfinite(hidden)).item():raise ValueError('Nonfinite reference states')
                accepted.append({'id':case['id'],'modality':case['modality'],'tokens':inputs.shape[1]})
                token_count+=inputs.shape[1];durations.append(time.monotonic()-case_start)
                event('calibration_case',**accepted[-1],seconds=durations[-1])
                del hidden,embeddings,inputs,media;gc.collect();mx.clear_cache()
                if len(accepted)%6==0:stats.save(root/'statistics',accepted)
            except Exception as error:
                errors.append({'id':case['id'],'phase':'calibration','error':repr(error)})
                event('calibration_case_failed',**errors[-1])
                if len(errors)>=3:break
        coverage=stats.save(root/'statistics',accepted)
        stats.enabled=False
        counts=collections.Counter(r['modality'] for r in accepted)
        enough=counts['text']>=24 and counts['image']>=8
        report.update(calibration_cases=dict(counts),calibration_tokens=token_count,
            calibration_sufficient=enough,load_seconds=load_seconds,
            calibration_seconds=time.monotonic()-calibration_start)
        # Fix a balanced comparison size from measured source throughput before candidate scoring.
        development=read_json(data/'development.json')
        average=sum(durations)/max(1,len(durations))
        target=max(18,min(72,int(4*3600/max(average*8,1))))
        target=target//3*3
        atomic_json(root/'DEVELOPMENT_SUBSET.json',{'ids':[r['id'] for r in development[:target]],
            'reason':'Fixed from pilot throughput before candidate scores','target_cases':target,
            'source_calibration_seconds_per_case':average})
        for case in development[:target]:
            if time.monotonic()-started>args.seconds-30:break
            t=time.monotonic()
            try:
                row=score_case(model,config,tokenizer,case,data);row['seconds']=time.monotonic()-t
                records.append(row)
                event('reference_score',**row)
            except Exception as error:
                errors.append({'id':case['id'],'phase':'reference_scoring','error':repr(error)})
            report.update(records=records,errors=errors,metrics=aggregate(records))
            atomic_json(root/'reference_metrics.json',report)
            gc.collect();mx.clear_cache()
        # Only compare cases whose reference likelihood is actually available.
        atomic_json(root/'DEVELOPMENT_SUBSET.json',{'ids':[r['id'] for r in records],
            'reason':'Fixed completed source subset before candidate scoring','target_cases':target,
            'complete_cases':len(records)})
        report['status']='complete' if enough and len(records)>=18 else 'reduced'
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        try:stats.save(root/'statistics',accepted)
        except Exception:pass
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,records=records,errors=errors,
            metrics=aggregate(records))
        atomic_json(root/'reference_metrics.json',report)
    event('reference_phase_finished',status=report['status'],cases=len(records))

def selected_statistics(root,path,expert):
    coverage=read_json(root/'statistics'/'coverage.json',{})
    item=coverage.get('modules',{}).get(path)
    if not item:return None,0
    with np.load(root/'statistics'/item['file'],allow_pickle=False) as data:
        count=int(data['count'][expert])
        return (data['sum2'][expert]/max(count,1) if count>=32 else None),count

def produce_search(args):
    root=args.run;source=Path(args.source);start=time.monotonic()
    model,config=load_model(source,lazy=True,strict=True)
    layout=[];unchanged_bytes=0
    for path,module in model.named_modules():
        if isinstance(module,MATRIX_CLASSES):
            shape=matrix_dims(module)
            layout.append({'path':path,'shape':list(shape),'default_bits':choose_bits(path,module),
                'target_metadata_bytes_per_group':8 if '.mlp.experts.' in path and getattr(module,'mode',None)=='nvfp4' and module.get('global_scale') is not None else 4,
                'source_bytes':sum(v.nbytes for v in module.parameters().values() if isinstance(v,mx.array))})
    atomic_json(root/'MODEL_LAYOUT.json',{'modules':layout,
        'parameter_bytes':sum(v.nbytes for _,v in __import__('mlx.utils',fromlist=['tree_flatten']).tree_flatten(model.parameters()))})
    modules=expert_modules(model);completed=0;total=sum(m.num_experts for _,m in modules)
    deadline=time.monotonic()+args.seconds-60
    use_gpu=args.gpu
    for path,module in modules:
        for expert in range(module.num_experts):
            if time.monotonic()>deadline:raise TimeoutError('Search time budget exhausted')
            identity=tensor_id(path,expert);result=root/'search'/(identity+'.json')
            if result.exists():completed+=1;continue
            statistic,count=selected_statistics(root,path,expert)
            weights=reconstruct_expert(module,expert)
            mx.eval(weights)
            if use_gpu:
                # Bound the pending payloads to roughly a few hundred MiB.
                while len(list((root/'queue').glob('*.npz')))>=24:
                    if time.monotonic()>deadline:raise TimeoutError('GPU queue time budget exhausted')
                    failures=list((root/'search').glob('*.error.json'))
                    if failures:raise RuntimeError('GPU search task failed: '+str(failures[0]))
                    time.sleep(2)
                payload=identity+'.npz';temporary=root/'queue'/(payload+'.tmp')
                with temporary.open('wb') as f:
                    np.savez(f,weights=np.asarray(weights.astype(mx.float32)),
                        statistic=np.zeros(weights.shape[-1],np.float32) if statistic is None else statistic)
                temporary.replace(root/'queue'/payload)
                task={'id':identity,'path':path,'expert':expert,'payload':payload,'weighted':statistic is not None,
                    'metadata_dtype':'float32' if weights.dtype==mx.float32 else 'bfloat16',
                    'routed_tokens':count,'shape':list(weights.shape),'run_identity':read_json(root/'RUN_MANIFEST.json')['run_id']}
                atomic_json(root/'queue'/(identity+'.ready.json'),task)
            else:
                # Exact MLX fallback uses the smaller three factor grid.
                maps,metrics=cpu_search(weights,statistic,alphas=[1.,.95,.90])
                # Store indices in the canonical 11 factor index space.
                lookup=np.array([0,4,6],np.uint8)
                maps={k:lookup[v] for k,v in maps.items()}
                with (root/'search'/(identity+'.npz')).open('wb') as f:np.savez_compressed(f,**maps)
                atomic_json(result,{'status':'complete','task':{'id':identity,'path':path,'expert':expert,
                    'weighted':statistic is not None,'routed_tokens':count,'shape':list(weights.shape)},
                    'metrics':metrics,'worker':'MLX CPU fallback'})
            completed+=1
            if completed%64==0:
                event('search_produced',completed=completed,total=total)
                atomic_json(root/'search_progress.json',{'produced':completed,'total':total,'seconds':time.monotonic()-start})
            del weights;mx.clear_cache()
    (root/'PRODUCER_DONE').touch()
    while len(list((root/'search').glob('*.json')))-len(list((root/'search').glob('*.error.json')))<total:
        if time.monotonic()>deadline:raise TimeoutError('Waiting for final search results exceeded budget')
        failures=list((root/'search').glob('*.error.json'))
        if failures:raise RuntimeError('GPU search task failed: '+str(failures[0]))
        time.sleep(2)
    atomic_json(root/'SEARCH_COMPLETE.json',{'expert_matrices':total,'seconds':time.monotonic()-start,'gpu':use_gpu})

def screen(args):
    root=args.run;manifest=read_json(root/'RUN_MANIFEST.json');layout=read_json(root/'MODEL_LAYOUT.json')
    records=[read_json(p) for p in sorted((root/'search').glob('*.json')) if not p.name.endswith('.error.json')]
    ready=(root/'SEARCH_COMPLETE.json').exists()
    weighted=read_json(root/'reference_metrics.json',{}).get('calibration_sufficient',False)
    costs=collections.defaultdict(float);layer_costs=collections.defaultdict(float)
    for result in records:
        task=result['task'];path=task['path'];layer=layer_number(path)
        for key,value in result['metrics'].items():costs[(path,key)]+=value
        # Sensitivity is the benefit of protecting a layer at 4 bits, per byte added.
        key='W_g128_b2' if weighted else 'C_g128_b2'
        key4='W_g128_b4' if weighted else 'C_g128_b4'
        visit_count=3 if 6<=layer<18 else 1
        layer_costs[layer]+=max(0,result['metrics'][key]-result['metrics'][key4])*visit_count
    protected=[x[0] for x in sorted(layer_costs.items(),key=lambda x:x[1],reverse=True)[:2]]
    selected=[]
    for original in manifest['recipes']:
        recipe=dict(original);method=recipe['method'];group=recipe['group_size']
        if method!='R' and not ready:continue
        if method=='W' and not weighted:continue
        if isinstance(recipe['expert_layers_at_4_bits'],str):recipe['expert_layers_at_4_bits']=protected
        total=layout['parameter_bytes'];error=0.
        for mod in layout['modules']:
            path=mod['path'];shape=mod['shape'];bits=mod['default_bits']
            if '.mlp.experts.' in path:bits=4 if layer_number(path) in recipe['expert_layers_at_4_bits'] else 2
            elif path=='lm_head' or path.endswith('.embed_tokens'):bits=recipe['embedding_and_head_bits']
            elif '.self_attn.' in path and bits is not None:bits=recipe['attention_bits']
            if bits is None or shape[-1]%group:continue
            n=int(np.prod(shape));new=n*bits//8+n//group*mod['target_metadata_bytes_per_group']
            total+=new-mod['source_bytes']
            if '.mlp.experts.' in path:
                key=f'{method}_g{group}_b{bits}';error+=costs[(path,key)]
        recipe.update(predicted_weight_bytes=total+2**20,screen_error=error,
            screen_objective='activation diagonal weighted error' if method=='W' else 'weight reconstruction error')
        selected.append(recipe)
    atomic_json(root/'SCREENED_RECIPES.json',{'recipes':selected,'protected_layers':protected,
        'complete_search':ready,'calibration_sufficient':weighted,
        'screening_does_not_establish_accuracy':True})
    event('recipes_screened',count=len(selected),protected_layers=protected)

def convert_phase(args):
    from convert_inkling import convert
    root=args.run;recipes=read_json(root/'SCREENED_RECIPES.json')['recipes']
    recipe=next(r for r in recipes if r['id']==args.candidate)
    report=convert(Path(args.source),root/'candidates'/args.candidate,recipe['group_size'],
        recipe=recipe,search_dir=root/'search',checkpoint_dir=root/'tensor-checkpoints'/args.candidate)
    (root/'exports').mkdir(exist_ok=True)
    shutil.copy2(root/'candidates'/args.candidate/'CONVERSION_REPORT.json',root/'exports'/(args.candidate+'-CONVERSION_REPORT.json'))
    # Successful reload makes the temporary module checkpoint files expendable.
    shutil.rmtree(root/'tensor-checkpoints'/args.candidate)
    event('candidate_export_verified',candidate=args.candidate,weight_bytes=report['weight_bytes'])

def evaluate(args):
    root=args.run;data=KIT/'data';output=root/'evaluation'/(args.candidate+'.json')
    subset=read_json(root/'DEVELOPMENT_SUBSET.json')
    ids=subset['ids'];definitions={c['id']:c for c in read_json(data/'development.json')}
    if args.limit_cases:ids=ids[:args.limit_cases]
    if not ids:raise ValueError('No fixed development comparison cases')
    context=digest({'model_report':sha_file(Path(args.model)/'CONVERSION_REPORT.json'),
        'ids':ids,'data':sha_file(data/'DATA_MANIFEST.json'),'cpu_dtype':'FP32'})
    conversion=read_json(Path(args.model)/'CONVERSION_REPORT.json')
    prior=read_json(output,{})
    records=prior.get('records',[]) if prior.get('context_hash')==context else []
    if prior.get('status')=='complete' and prior.get('context_hash')==context and len(records)==len(ids):return
    completed={r['id'] for r in records};started=time.monotonic();report={}
    try:
        model,config=load_candidate(args.model);tokenizer=load_tokenizer(Path(args.model))
        for identity in ids:
            if identity in completed:continue
            t=time.monotonic();row=score_case(model,config,tokenizer,definitions[identity],data)
            row['seconds']=time.monotonic()-t;records.append(row)
            report={'status':'running','candidate':args.candidate,'records':records,'metrics':aggregate(records),
                'context_hash':context,'expected_cases':len(ids),'cpu_arithmetic':'FP32 with CUDA matrix operators'}
            atomic_json(output,report);event('candidate_score',candidate=args.candidate,**row)
            gc.collect();mx.clear_cache()
        report['status']='complete'
    except BaseException as error:
        report.update(status='incomplete',error=repr(error),traceback=traceback.format_exc())
    finally:
        report.update(candidate=args.candidate,records=records,metrics=aggregate(records),
            context_hash=context,expected_cases=len(ids),elapsed_seconds=time.monotonic()-started,
            weight_bytes=conversion['weight_bytes'],recipe=conversion.get('recipe'),
            weight_gib=conversion['weight_bytes']/2**30)
        atomic_json(output,report)
    if report['status']!='complete':raise RuntimeError('Candidate evaluation incomplete')

def generate_smoke(args):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    root=args.run;data=KIT/'data';path=root/'generation'/(args.candidate+'.json')
    report={'candidate':args.candidate,'status':'running','cases':[],'max_new_tokens':192,
        'scope':'Bounded CPU smoke examples, separate from teacher forced ranking'}
    try:
        model,config=load_candidate(args.model);tokenizer=load_tokenizer(Path(args.model))
        cases=read_json(data/'development.json')[:args.max_cases]
        for case in cases:
            inputs,media,_,_=case_arrays(case,data,tokenizer,config,with_answer=False)
            tokens=inputs[0];embeddings=model.embed_inputs(tokens,**media) if media else None
            if embeddings is not None:mx.eval(embeddings)
            started=time.monotonic();pieces=[];final=None
            for response in stream_generate(model,tokenizer,tokens,max_tokens=192,sampler=make_sampler(temp=0),
                    prefill_step_size=64,input_embeddings=embeddings):
                pieces.append(response.text);final=response
            row={'id':case['id'],'modality':case['modality'],'expected':case['answer'],
                'output':''.join(pieces),'seconds':time.monotonic()-started,
                'generation_tokens':final.generation_tokens if final else 0,
                'generation_tps':final.generation_tps if final else None}
            report['cases'].append(row);atomic_json(path,report);event('generation_case',candidate=args.candidate,**row)
            del embeddings,inputs,tokens;gc.collect();mx.clear_cache()
        report['status']='complete'
    except BaseException as error:
        report.update(status='incomplete',error=repr(error))
    finally:atomic_json(path,report)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pilot','reference','produce-search','screen','convert','evaluate','generate'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--source',default='/work/source')
    parser.add_argument('--model');parser.add_argument('--candidate');parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--seconds',type=int,default=10800);parser.add_argument('--max-cases',type=int,default=3)
    parser.add_argument('--limit-cases',type=int,default=0)
    args=parser.parse_args();install_deadline(args.seconds);check_runtime();mx.set_default_device(mx.cpu)
    mx.set_cache_limit(256*1024**2)
    {'pilot':pilot,'reference':reference,'produce-search':produce_search,'screen':screen,'convert':convert_phase,
        'evaluate':evaluate,'generate':generate_smoke}[args.phase](args)

if __name__=='__main__':main()
