"""Local handoff: wait for Leonardo, fetch verified finalists, then test on Metal."""
from __future__ import annotations
import argparse, json, os, random, re, shlex, shutil, subprocess, sys, time, traceback
from pathlib import Path
from common import atomic_json,read_json,sha_file

ROOT=Path(__file__).resolve().parents[2]
KIT=Path(__file__).resolve().parent

def system_memory():
    values={}
    for key in ['vm.swapusage','kern.memorystatus_vm_pressure_level']:
        p=subprocess.run(['sysctl','-n',key],capture_output=True,text=True)
        values[key]=p.stdout.strip()
    return values

def guarded_process(command,report,log,seconds):
    start=time.monotonic();critical=None;reason=None
    with Path(log).open('a') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        while process.poll() is None:
            state=system_memory()
            if state['kern.memorystatus_vm_pressure_level']=='4':critical=critical or time.monotonic()
            else:critical=None
            if critical and time.monotonic()-critical>20:reason='Critical system memory pressure for 20 seconds'
            if time.monotonic()-start>seconds:reason='Per model test time limit reached'
            if reason:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(3)
    if reason:
        record=read_json(report,{})
        record.update(status='stopped_by_resource_guard',watchdog=reason)
        atomic_json(report,record)
    return {'returncode':process.returncode,'seconds':time.monotonic()-start,'guard':reason}

def comparison(output):
    files=sorted((output/'mac').glob('heldout-*.json'))
    reports={read_json(p)['candidate']:read_json(p) for p in files}
    baseline=reports.get('R64',{})
    by_id={r['id']:r for r in baseline.get('cases',[])}
    rows=[]
    for name,report in reports.items():
        if name=='R64':continue
        matched=[r for r in report.get('cases',[]) if r['id'] in by_id]
        if not matched:continue
        difference=sum(r['score']-by_id[r['id']]['score'] for r in matched)/len(matched)
        changes=[r['score']-by_id[r['id']]['score'] for r in matched]
        rng=random.Random(20261121)
        boot=sorted(sum(rng.choice(changes) for _ in changes)/len(changes) for _ in range(2000))
        categories={}
        for group in sorted({r['category'] for r in matched}):
            chosen=[r for r in matched if r['category']==group]
            categories[group]={'cases':len(chosen),'paired_difference':sum(r['score']-by_id[r['id']]['score'] for r in chosen)/len(chosen)}
        complete=len(matched)==180 and baseline.get('status')=='complete' and report.get('status')=='complete'
        passed=complete and difference>=-.02 and all(v['paired_difference']>=-.05-1e-9 for v in categories.values())
        rows.append({'candidate':name,'paired_cases':len(matched),'paired_accuracy_difference':difference,
            'bootstrap_interval_95':[boot[49],boot[1949]],'categories':categories,
            'all_180_cases_completed':complete,'passes_engineering_quality_filter':passed})
    result={'baseline':'R64','comparisons':rows,'reports':{k:v.get('summary',{}) for k,v in reports.items()},
        'scope':'Frozen custom diagnostic subset; uncertainty intervals do not prove equivalence',
        'memory_and_context_reports_are_separate':True}
    atomic_json(output/'MAC_COMPARISON.json',result)
    lines=['Inkling series 2 overnight results',f'Leonardo results: {output / "leonardo"}',
        f'Mac heldout reports: {output / "mac"}',f'Paired comparisons: {len(rows)}']
    for row in rows:
        lines.append(f'{row["candidate"]}: {row["paired_cases"]} paired cases; quality change {100*row["paired_accuracy_difference"]:+.2f} percentage points; quality filter {row["passes_engineering_quality_filter"]}.')
    lines.append('Read memory pressure, swap and context timings before choosing a deployment model.')
    (output/'EXPERIMENT_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,required=True);a=p.parse_args()
    meta=read_json(a.metadata);output=ROOT/'outputs/series2-results'/meta['id']
    output.mkdir(parents=True,exist_ok=True);(output/'mac').mkdir(exist_ok=True)
    status_path=output/'LOCAL_STATUS.json'
    ssh=['ssh','-S',str(ROOT/'work/s'),'-o','BatchMode=yes','-o','ConnectTimeout=15','leonardo']
    status={'status':'waiting_for_leonardo','pid':os.getpid(),'run_id':meta['id'],'started_unix':time.time()}
    atomic_json(status_path,status)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
    started=time.monotonic();failed_since=None
    try:
        while time.monotonic()-started<18*3600:
            remote=subprocess.run(ssh+['cat '+shlex.quote(meta['run']+'/STATUS.json')],capture_output=True,text=True,timeout=40)
            if remote.returncode:
                failed_since=failed_since or time.monotonic()
                status.update(last_ssh_error=remote.stderr[-1500:])
                if time.monotonic()-failed_since>1800:raise RuntimeError('SSH unavailable for 30 minutes; Leonardo job may still be running')
            else:
                failed_since=None;remote_status=json.loads(remote.stdout)
                status.update(leonardo=remote_status,last_poll_unix=time.time());atomic_json(status_path,status)
                if remote_status.get('status','').startswith('complete'):break
                if remote_status.get('status')=='failed':raise RuntimeError('Leonardo controller failed; inspect its logs')
            time.sleep(120)
        else:raise TimeoutError('Local handoff exceeded 18 hours')
        archive=remote_status['archive']
        expected=meta['project']+'/experiments/'+meta['id']
        if archive!=expected:raise ValueError('Unexpected experiment archive path')
        status.update(status='downloading_results');atomic_json(status_path,status)
        ssh_transport=shlex.join(ssh[:-1])
        subprocess.run(['rsync','-az','--partial','--exclude=finalists/','--exclude=search/','-e',ssh_transport,
            'leonardo:'+archive+'/',str(output/'leonardo')+'/'],check=True,timeout=1800)
        finalists=read_json(output/'leonardo/FINALISTS.json',[])
        if not finalists:raise RuntimeError('No verified finalists were produced')
        baseline_bytes=sum(p.stat().st_size for p in (ROOT/'outputs/Inkling-2-Mini-expert-2bit-g64').glob('model*.safetensors'))
        eligible=[r for r in finalists if r['weight_bytes']<=baseline_bytes]
        if not eligible:raise RuntimeError('No finalist is as small as the existing 16 GiB Mac baseline')
        smallest=min(eligible,key=lambda r:r['weight_bytes']);quality=min(eligible,key=lambda r:r['balanced_nll'])
        chosen=list({r['candidate']:r for r in [smallest,quality]}.values())
        local_models=[]
        for row in chosen:
            name=row['candidate']
            if not re.fullmatch(r'[RCW][A-Za-z0-9-]+',name):raise ValueError('Unsafe candidate name')
            if row['path']!=archive+'/finalists/'+name:raise ValueError('Unexpected finalist directory')
            if shutil.disk_usage(ROOT).free<row['weight_bytes']+20*2**30:raise RuntimeError('Insufficient local free space for finalist plus reserve')
            target=output/'models'/name;target.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-az','--partial','-e',ssh_transport,'leonardo:'+row['path']+'/',str(target)+'/'],check=True,timeout=7200)
            for relative,expected_hash in row['file_hashes'].items():
                part=Path(relative)
                if part.is_absolute() or '..' in part.parts:raise ValueError('Unsafe file hash entry')
                if sha_file(target/part)!=expected_hash:raise ValueError('Downloaded finalist checksum mismatch')
            local_models.append((name,target))
        status.update(status='testing_on_mac',downloaded=[n for n,_ in local_models]);atomic_json(status_path,status)
        models=[('R64',ROOT/'outputs/Inkling-2-Mini-expert-2bit-g64')]+local_models
        native=ROOT/'work/validation-env/bin/python'
        mac_deadline=time.monotonic()+4*3600
        for name,model in models:
            if mac_deadline-time.monotonic()<180:break
            status.update(current_candidate=name);atomic_json(status_path,status)
            report=output/'mac'/('heldout-'+name+'.json')
            command=[str(native),'-u',str(KIT/'mac_evaluate.py'),'--model',str(model),
                '--candidate',name,'--report',str(report)]
            result=guarded_process(command,report,output/'mac'/('heldout-'+name+'.log'),min(5400,mac_deadline-time.monotonic()))
            status.setdefault('mac_runs',{})[name]=result;atomic_json(status_path,status)
        # The default launcher has a 4096 total token cap; reserve 128 tokens for output.
        for name,model in [models[0],local_models[0]]:
            for prefill in [64,256]:
                for repeat in range(3):
                    if mac_deadline-time.monotonic()<120:break
                    report=output/'mac'/f'context-{name}-p{prefill}-r{repeat}.json'
                    command=[str(native),'-u',str(ROOT/'outputs/mac_probe.py'),'--model',str(model),
                        '--report',str(report),'--cases','','--context-lengths','512,2048,3968',
                        '--prefill-step',str(prefill),'--max-tokens','128']
                    result=guarded_process(command,report,report.with_suffix('.log'),min(900,mac_deadline-time.monotonic()))
                    if result['guard']:break
        result=comparison(output)
        full=all(read_json(output/'mac'/('heldout-'+name+'.json'),{}).get('status')=='complete' for name,_ in models)
        status.update(status='complete' if full else 'complete_with_incomplete_mac_tests',finished_unix=time.time(),comparison_count=len(result['comparisons']))
        atomic_json(status_path,status)
    except BaseException as error:
        status.update(status='needs_attention',error=repr(error),traceback=traceback.format_exc(),updated_unix=time.time())
        atomic_json(status_path,status);raise
    finally:
        awake.terminate()

if __name__=='__main__':main()
