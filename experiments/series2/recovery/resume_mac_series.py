"""Resume unchanged heldout tests after Metal timeouts, within original budget."""
import json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
KIT=ROOT/'outputs/series2-kit';sys.path.insert(0,str(KIT))
from common import atomic_json,read_json
from watch_overnight import guarded_process,comparison

def main():
    meta_path=ROOT/'work/series2-run.json';meta=read_json(meta_path)
    out=ROOT/'outputs/series2-results'/meta['id'];status_path=out/'LOCAL_STATUS.json'
    prior=read_json(status_path);deadline=(out/'mac/heldout-R64.json').stat().st_birthtime+4*3600
    status={**prior,'status':'testing_on_mac','phase':'resuming_after_metal_timeout','pid':os.getpid(),
        'recovery_started_unix':time.time(),'mac_stage_deadline_unix':deadline}
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
    atomic_json(status_path,status)
    try:
        old=meta['local_watcher_pid']
        old_cmd=subprocess.run(['ps','-p',str(old),'-o','command='],capture_output=True,text=True).stdout
        if 'watch_overnight.py' in old_cmd:os.kill(old,signal.SIGTERM)
        meta.update(local_watcher_previous_pid=old,local_watcher_pid=os.getpid(),
            local_handoff_script=str(Path(__file__).resolve()))
        atomic_json(meta_path,meta)
        native=ROOT/'work/validation-env/bin/python'
        for name in ['R64','W128-S4']:
            if deadline-time.time()<180:break
            model=ROOT/'outputs/Inkling-2-Mini-expert-2bit-g64' if name=='R64' else out/'models'/name
            report=out/'mac'/f'heldout-{name}.json'
            status.update(current_candidate=name,phase='resuming_heldout',updated_unix=time.time());atomic_json(status_path,status)
            command=[str(native),'-u',str(KIT/'mac_evaluate.py'),'--model',str(model),
                '--candidate',name,'--report',str(report)]
            result=guarded_process(command,report,out/'mac'/f'heldout-{name}-resume.log',min(1800,deadline-time.time()))
            status.setdefault('resumed_mac_runs',{})[name]=result
            atomic_json(status_path,status)
        # Retain interrupted context evidence; do one bounded repeat per model.
        for name in ['R64','W128-E4']:
            if deadline-time.time()<180:break
            model=ROOT/'outputs/Inkling-2-Mini-expert-2bit-g64' if name=='R64' else out/'models'/name
            report=out/'mac'/f'context-recovery-{name}-p64.json'
            status.update(current_candidate=name,phase='bounded_context_recheck',updated_unix=time.time());atomic_json(status_path,status)
            command=[str(native),'-u',str(ROOT/'outputs/mac_probe.py'),'--model',str(model),
                '--report',str(report),'--cases','','--context-lengths','512,2048,3968',
                '--prefill-step','64','--max-tokens','128']
            result=guarded_process(command,report,report.with_suffix('.log'),min(600,deadline-time.time()))
            status.setdefault('context_rechecks',{})[name]=result;atomic_json(status_path,status)
        result=comparison(out)
        # The extraction prompts ask for a shipment code but only label an order.
        # Keep original frozen scores and add a clearly labelled sensitivity view.
        sensitivity={}
        for name in ['R64','W128-E4','W128-S4']:
            report=read_json(out/'mac'/f'heldout-{name}.json',{})
            rows=[r for r in report.get('cases',[]) if r['category']!='extraction']
            sensitivity[name]={'cases':len(rows),'score':sum(r['score'] for r in rows)/len(rows) if rows else None,
                'excluded_category':'extraction','excluded_cases':sum(r['category']=='extraction' for r in report.get('cases',[])),
                'original_frozen_report_unchanged':True}
        atomic_json(out/'EXTRACTION_SENSITIVITY.json',{'reason':'Ambiguous authored prompts request a shipment code but explicitly identify only order and warehouse codes.',
            'analysis_is_post_hoc':True,'reports':sensitivity})
        complete=all(read_json(out/'mac'/f'heldout-{n}.json',{}).get('status')=='complete' for n in ['R64','W128-E4','W128-S4'])
        status.update(status='complete' if complete else 'complete_with_incomplete_mac_tests',phase='finished',
            finished_unix=time.time(),comparison_count=len(result['comparisons']),
            scope_note='Context repeats reduced after Metal timeouts; extraction ambiguity separately disclosed.')
        atomic_json(status_path,status)
    except BaseException as error:
        status.update(status='needs_attention',error=repr(error),traceback=traceback.format_exc(),updated_unix=time.time())
        atomic_json(status_path,status);raise
    finally:awake.terminate()

if __name__=='__main__':main()
