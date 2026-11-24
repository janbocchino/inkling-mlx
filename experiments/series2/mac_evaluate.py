"""Bounded native Metal evaluation on the frozen heldout questions."""
from __future__ import annotations
import os
os.environ.setdefault('MLX_ENABLE_TF32','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import argparse, collections, gc, json, re, resource, subprocess, time, unicodedata
from pathlib import Path
import mlx.core as mx
from mlx_lm import load,stream_generate
from mlx_lm.sample_utils import make_sampler
from model_tools import case_arrays
from common import atomic_json,read_json,digest,sha_file,event

def memory_state():
    result={}
    for key in ['vm.swapusage','kern.memorystatus_vm_pressure_level']:
        p=subprocess.run(['sysctl','-n',key],capture_output=True,text=True)
        result[key]=p.stdout.strip() if p.returncode==0 else None
    return result

def final_text(text):
    if '<|content_text|>' not in text:return '',False
    answer=text.rsplit('<|content_text|>',1)[1]
    return answer.split('<|end_message|>',1)[0].strip(),'<|end_message|>' in answer

def normalize(text):
    text=unicodedata.normalize('NFKC',str(text)).lower().strip()
    text=re.sub(r'[^\w\s.\-]',' ',text)
    text=re.sub(r'\b(a|an|the)\b',' ',text)
    numbers=dict(zip('zero one two three four five six seven eight nine ten'.split(),map(str,range(11))))
    return ' '.join(numbers.get(x,x) for x in text.split()).strip().rstrip('.')

def correctness(case,answer,has_final):
    if not has_final:return 0.
    if case['category']=='structured':
        try:return float(json.loads(answer.strip().removeprefix('```json').removeprefix('```').removesuffix('```'))==json.loads(case['answer']))
        except (ValueError,TypeError):return 0.
    if case['category']=='chart':
        try:
            target=float(case['answer'].replace(',','').replace('%',''))
            value=float(answer.replace(',','').replace('%',''))
            return float(abs(value-target)<=abs(target)*.05 if target else value==0)
        except ValueError:return float(normalize(answer)==normalize(case['answer']))
    if case['category']=='scene_text':
        answers=[normalize(x) for x in case['answers']];prediction=normalize(answer)
        if len(answers)<2:return float(prediction in answers)
        return sum(min(sum(prediction==a for j,a in enumerate(answers) if j!=i)/3,1) for i in range(len(answers)))/len(answers)
    return float(normalize(answer)==normalize(case['answer']))

def summary(rows):
    result={}
    for category in ['all','text','image']+sorted({r['category'] for r in rows}):
        chosen=[r for r in rows if category=='all' or r['modality']==category or r['category']==category]
        if chosen:result[category]={'cases':len(chosen),'score':sum(r['score'] for r in chosen)/len(chosen),
            'missing_final_answers':sum(not r['has_final'] for r in chosen)}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--limit',type=int,default=180);a=p.parse_args()
    data=Path(__file__).resolve().parent/'data';cases=read_json(data/'heldout.json')[:a.limit]
    context=digest({'model':sha_file(a.model/'CONVERSION_REPORT.json'),'data':sha_file(data/'DATA_MANIFEST.json'),
        'ids':[c['id'] for c in cases],'max_new_tokens':192,'prefill':64})
    prior=read_json(a.report,{})
    rows=prior.get('cases',[]) if prior.get('context_hash')==context else []
    if prior.get('status')=='complete' and len(rows)==len(cases):return
    completed={r['id'] for r in rows}
    report={'status':'running','candidate':a.candidate,'context_hash':context,'cases':rows,
        'hardware':mx.device_info(),'system_before':memory_state(),'max_new_tokens':192,
        'score_scope':'Custom diagnostic subset with short answers, approximate VQA normalization and chart 5 percent tolerance; not official benchmark scores'}
    mx.set_memory_limit(int(mx.device_info()['max_recommended_working_set_size']*.98))
    mx.set_cache_limit(128*1024**2)
    started=time.monotonic();atomic_json(a.report,report)
    try:
        model,tokenizer,config=load(str(a.model),return_config=True)
        report['load_seconds']=time.monotonic()-started
        # One unscored warmup uses calibration data, keeping heldout answers untouched.
        warmup=next(c for c in read_json(data/'calibration.json') if c['answer'] and c['modality']=='text')
        inputs,_,_,_=case_arrays(warmup,data,tokenizer,config,with_answer=False)
        for _ in stream_generate(model,tokenizer,inputs[0],max_tokens=8,sampler=make_sampler(temp=0),prefill_step_size=64):pass
        for case in cases:
            if case['id'] in completed:continue
            inputs,media,_,_=case_arrays(case,data,tokenizer,config,with_answer=False)
            tokens=inputs[0];mx.clear_cache();mx.reset_peak_memory()
            t=time.monotonic();first=None;fragments=[];last=None
            embeddings=model.embed_inputs(tokens,**media) if media else None
            if embeddings is not None:mx.eval(embeddings)
            for response in stream_generate(model,tokenizer,tokens,max_tokens=192,sampler=make_sampler(temp=0),
                    prefill_step_size=64,input_embeddings=embeddings):
                if first is None:first=time.monotonic()-t
                fragments.append(response.text);last=response
            raw=''.join(fragments);answer,has_final=final_text(raw)
            row={'id':case['id'],'modality':case['modality'],'category':case['category'],
                'answer':answer,'has_final':has_final,'raw_output':raw,'expected':case['answer'],
                'score':correctness(case,answer,has_final),'seconds':time.monotonic()-t,
                'first_token_seconds':first,'peak_mlx_bytes':mx.get_peak_memory(),
                'prompt_tokens':last.prompt_tokens if last else 0,'generation_tokens':last.generation_tokens if last else 0,
                'generation_tps':last.generation_tps if last else None,'system_after':memory_state()}
            rows.append(row);report.update(cases=rows,summary=summary(rows),elapsed_seconds=time.monotonic()-started)
            atomic_json(a.report,report);event('mac_heldout_case',candidate=a.candidate,id=case['id'],score=row['score'])
            del embeddings,inputs,tokens,media;gc.collect();mx.clear_cache()
        report['status']='complete'
    except BaseException as error:
        report.update(status='incomplete',error=repr(error));raise
    finally:
        report.update(cases=rows,summary=summary(rows),elapsed_seconds=time.monotonic()-started,system_after=memory_state())
        atomic_json(a.report,report)

if __name__=='__main__':main()
