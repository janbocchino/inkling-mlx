"""Durable Slurm controller with bounded phases, recovery records and final export."""
from __future__ import annotations
import argparse, csv, datetime, json, os, re, shutil, signal, subprocess, sys, time, traceback
from zoneinfo import ZoneInfo
from pathlib import Path
from common import atomic_json, read_json, sha_file, digest, directory_bytes, event

class Controller:
    def __init__(self,project,root,kit):
        self.project,self.root,self.kit=Path(project),Path(root),Path(kit)
        self.manifest=read_json(root/'RUN_MANIFEST.json')
        self.archive=self.project/'experiments'/self.manifest['run_id']
        self.archive.mkdir(parents=True,exist_ok=True)
        prior=read_json(root/'STATUS.json',{})
        self.started=prior.get('start_unix',time.time());self.deadline=self.started+12*3600
        service_job=self.manifest.get('matrix_service_job')
        if service_job:
            info=subprocess.run(['scontrol','show','job',str(service_job)],capture_output=True,text=True)
            match=re.search(r'EndTime=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)',info.stdout)
            if match:
                end=datetime.datetime.fromisoformat(match.group(1)).replace(tzinfo=ZoneInfo('Europe/Vienna')).timestamp()
                self.deadline=min(self.deadline,end-120)
        self.worker_ids=[];self.stop=False;self.phase='starting';self.failures=[]
        self.state={'status':'running','run_id':self.manifest['run_id'],'controller_job':os.environ.get('SLURM_JOB_ID'),
            'start_unix':self.started,'deadline_unix':self.deadline,'archive':str(self.archive)}
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,self.request_stop)
    def request_stop(self,*_):self.stop=True
    def heartbeat(self):
        self.state.update(phase=self.phase,updated_unix=time.time(),worker_jobs=self.worker_ids,failures=self.failures)
        atomic_json(self.root/'heartbeat.json',{'unix_time':time.time(),'controller_job':self.state['controller_job']})
        atomic_json(self.root/'STATUS.json',self.state)
    def run_phase(self,phase,seconds,*arguments):
        if self.stop:return False
        seconds=min(int(seconds),int(self.deadline-time.time()-1800))
        if seconds<=0:return False
        self.phase=phase;self.heartbeat()
        for folder in ['logs','evaluation','generation','candidates','queue','search','tensor-checkpoints']:
            (self.root/folder).mkdir(parents=True,exist_ok=True)
        env='HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,MLX_ENABLE_TF32=0,OMP_NUM_THREADS=16,OPENBLAS_NUM_THREADS=16,PYTHONUNBUFFERED=1,SERIES_GPU_ENDPOINT=/run-series/GPU_ENDPOINT.json'
        cmd=['singularity','exec','--cleanenv','--bind',f'{self.project}:/work,{self.root}:/run-series',
            '--pwd','/work/series2-kit','--env',env,
            str(self.project/'containers/python311-bookworm.sif'),'/work/env-cpu/bin/python','-u',
            '/work/series2-kit/cpu_phases.py',*arguments,'--run','/run-series','--seconds',str(seconds)]
        log=self.root/'logs'/(phase+'.out');start=time.time()
        event('phase_started',phase=phase,budget_seconds=seconds)
        with log.open('a') as f:
            process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                self.heartbeat()
                if self.stop or time.time()-start>seconds+10:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                time.sleep(5)
        ok=process.returncode==0
        if not ok:self.failures.append({'phase':phase,'returncode':process.returncode,'log':str(log)})
        event('phase_finished',phase=phase,ok=ok,seconds=time.time()-start)
        return ok
    def workers(self):
        if (self.root/'SEARCH_COMPLETE.json').exists():return False
        (self.root/'STOP_WORKERS').unlink(missing_ok=True)
        (self.root/'PRODUCER_DONE').unlink(missing_ok=True)
        self.phase='waiting_for_gpu_workers';self.heartbeat()
        for _ in range(2):
            command=['sbatch','--parsable','--output='+str(self.root/'logs/gpu-%j.out'),
                '--error='+str(self.root/'logs/gpu-%j.err'),
                '--export=ALL,SERIES_KIT='+str(self.kit)+',SERIES_RUN='+str(self.root)+',PARITY_ONLY=0',
                str(self.kit/'gpu_worker.sbatch')]
            result=subprocess.run(command,capture_output=True,text=True)
            if result.returncode==0:self.worker_ids.append(result.stdout.strip().split(';')[0])
            else:self.failures.append({'phase':'gpu_submission','error':result.stderr})
        atomic_json(self.root/'WORKER_JOBS.json',self.worker_ids)
        started=time.time()
        while time.time()-started<1800 and self.worker_ids and not self.stop:
            self.heartbeat()
            results=[read_json(self.root/f'gpu-parity-{j}.json',{}) for j in self.worker_ids]
            if any(v.get('passed') for v in results):return True
            if all(v and not v.get('passed') for v in results):break
            time.sleep(10)
        self.cancel_workers();return False
    def cancel_workers(self):
        (self.root/'STOP_WORKERS').touch()
        for job in self.worker_ids:
            subprocess.run(['scancel',job],capture_output=True,text=True)
    def ensure_subset(self):
        subset=read_json(self.root/'DEVELOPMENT_SUBSET.json',{})
        if len(subset.get('ids',[]))>=18:return
        data=read_json(self.kit/'data/development.json')
        atomic_json(self.root/'DEVELOPMENT_SUBSET.json',{'ids':[c['id'] for c in data[:18]],
            'reason':'Reference phase incomplete; fixed minimum comparison subset',
            'source_reference_incomplete':True})
    def complete_evaluations(self):
        result=[]
        ids=read_json(self.root/'DEVELOPMENT_SUBSET.json')['ids']
        for p in (self.root/'evaluation').glob('*.json'):
            row=read_json(p)
            if row.get('status')=='complete' and [r['id'] for r in row['records']]==ids:
                result.append(row)
        return result
    def shortlist(self):
        rows=[r for r in self.complete_evaluations() if r['candidate']!='R64' and (self.root/'candidates'/r['candidate']).exists()]
        for row in rows:
            report=read_json(self.root/'candidates'/row['candidate']/'CONVERSION_REPORT.json')
            row['weight_bytes']=report['weight_bytes']
            row['balanced_nll']=.5*(row['metrics']['text']['nll_per_token']+row['metrics']['image']['nll_per_token'])
        if not rows:return []
        smallest=min(rows,key=lambda r:r['weight_bytes']);best=min(rows,key=lambda r:r['balanced_nll'])
        chosen={r['candidate'] for r in [smallest,best]}
        remaining=[r for r in rows if r['candidate'] not in chosen]
        if remaining:
            low_b=min(r['weight_bytes'] for r in rows);high_b=max(r['weight_bytes'] for r in rows)
            low_n=min(r['balanced_nll'] for r in rows);high_n=max(r['balanced_nll'] for r in rows)
            compromise=min(remaining,key=lambda r:(r['weight_bytes']-low_b)/max(high_b-low_b,1)+(r['balanced_nll']-low_n)/max(high_n-low_n,1e-9))
            chosen.add(compromise['candidate'])
        return [r for r in rows if r['candidate'] in chosen]
    def prune(self,needed_bytes=0):
        chosen=self.shortlist();keep={r['candidate'] for r in chosen}
        for folder in (self.root/'candidates').iterdir():
            if not folder.is_dir() or folder.name.endswith('.incomplete'):continue
            if folder.name not in keep:
                # Only this run's candidate files are expendable. Source and R64 are elsewhere.
                if folder.parent!=self.root/'candidates':raise ValueError('Unsafe cleanup path')
                shutil.rmtree(folder)
                event('discarded_export_after_scoring',candidate=folder.name)
        while directory_bytes(self.root)+directory_bytes(self.archive)+needed_bytes>60*2**30:
            choices=sorted(self.shortlist(),key=lambda r:r['balanced_nll'],reverse=True)
            if not choices:raise RuntimeError('Search cache and next candidate exceed storage budget')
            folder=self.root/'candidates'/choices[0]['candidate'];shutil.rmtree(folder)
            event('released_candidate_for_disk_budget',candidate=folder.name)
    def run(self):
        if read_json(self.root/'PILOT.json',{}).get('status')!='passed':
            raise RuntimeError('Full source pilot must pass before overnight run')
        if read_json(self.root/'HYBRID_PARITY.json',{}).get('status')!='passed':
            raise RuntimeError('Exact hybrid parity must pass before overnight run')
        for relative,expected in self.manifest.get('code_hashes',{}).items():
            if sha_file(self.kit/relative)!=expected:raise RuntimeError('Frozen code changed: '+relative)
        data_manifest=read_json(self.kit/'data/DATA_MANIFEST.json')
        for relative,expected in data_manifest['files'].items():
            if sha_file(self.kit/'data'/relative)!=expected:raise RuntimeError('Frozen data changed: '+relative)
        if not self.manifest.get('source_weight_sha256'):
            self.phase='source_fingerprint';self.heartbeat()
            self.manifest['source_weight_sha256']={p.name:sha_file(p) for p in (self.project/'source').glob('model*.safetensors')}
            atomic_json(self.root/'RUN_MANIFEST.json',self.manifest)
        if read_json(self.root/'reference_metrics.json',{}).get('status')!='complete':
            self.run_phase('source_reference_and_calibration',9000,'reference')
        self.ensure_subset()
        baseline='/work/'+str(self.manifest['baseline_relative'])
        if read_json(self.root/'evaluation/R64-PILOT.json',{}).get('status')!='complete':
            self.run_phase('baseline_timing',900,'evaluate','--candidate','R64-PILOT','--model',baseline,'--limit-cases','3')
        timing=read_json(self.root/'evaluation/R64-PILOT.json',{})
        seconds_per_case=timing.get('elapsed_seconds',900)/max(len(timing.get('records',[])),1)
        subset=read_json(self.root/'DEVELOPMENT_SUBSET.json')
        available=max(0,self.deadline-time.time()-3*3600-6*700-3600)
        target=max(18,min(len(subset['ids']),int(available/max(7*seconds_per_case,1))))//3*3
        subset['ids']=subset['ids'][:target];subset['baseline_pilot_seconds_per_case']=seconds_per_case
        subset['fixed_before_candidate_comparisons']=True
        atomic_json(self.root/'DEVELOPMENT_SUBSET.json',subset)
        if not (self.root/'SEARCH_COMPLETE.json').exists():
            use_gpu=self.workers()
            arguments=['produce-search']+(['--gpu'] if use_gpu else [])
            search_start=time.time()
            ok=self.run_phase('expert_clipping_search',10800,*arguments)
            if not ok and use_gpu and time.time()-search_start<10200:
                self.cancel_workers()
                self.run_phase('clipping_search_cpu_recovery',max(300,10800-(time.time()-search_start)),'produce-search')
        self.cancel_workers()
        if not (self.root/'MODEL_LAYOUT.json').exists():raise RuntimeError('Search did not produce source layout')
        self.run_phase('recipe_screening',300,'screen')
        self.run_phase('baseline_evaluation',max(900,seconds_per_case*len(subset['ids'])*1.5+180),
            'evaluate','--candidate','R64','--model',baseline)
        recipes={r['id']:r for r in read_json(self.root/'SCREENED_RECIPES.json')['recipes']}
        weighted='W128' in recipes
        preferred=['R128','C128', 'W128' if weighted else 'R32',
            ('W' if weighted else 'C')+'128-E4',('W' if weighted else 'C')+'128-A8',
            ('W' if weighted else 'C')+'128-S4']
        atomic_json(self.root/'PROMOTED_RECIPES.json',[recipes[name] for name in preferred if name in recipes])
        for name in preferred:
            if name not in recipes or self.stop or time.time()>min(self.started+10.5*3600,self.deadline-5400):continue
            recipe=recipes[name]
            if shutil.disk_usage(self.project).free<60*2**30:
                self.failures.append({'phase':'storage_guard','candidate':name,'error':'Work filesystem below reserve'});break
            try:self.prune(needed_bytes=2.2*recipe['predicted_weight_bytes'])
            except RuntimeError as e:
                self.failures.append({'phase':'storage_guard','candidate':name,'error':str(e)});continue
            output=self.root/'candidates'/name
            if not output.exists():
                ok=self.run_phase('export-'+name,2700,'convert','--candidate',name)
                if not ok:
                    # One retry reuses verified per-module checkpoints.
                    ok=self.run_phase('export-retry-'+name,2700,'convert','--candidate',name)
                if not ok:
                    for failed in [self.root/'tensor-checkpoints'/name,self.root/'candidates'/(name+'.incomplete')]:
                        if failed.exists():shutil.rmtree(failed)
                    continue
            model='/run-series/candidates/'+name
            budget=max(900,seconds_per_case*len(subset['ids'])*1.6+180)
            self.run_phase('evaluate-'+name,budget,'evaluate','--candidate',name,'--model',model)
            self.prune()
        shortlist=self.shortlist()
        remaining=max(0,self.deadline-time.time()-1800)
        per_model=min(1800,remaining/max(1,len(shortlist)))
        for row in shortlist:
            if per_model<180:break
            self.run_phase('smoke-'+row['candidate'],per_model,'generate','--candidate',row['candidate'],
                '--model','/run-series/candidates/'+row['candidate'],'--max-cases','3')
        self.finalize()
    def finalize(self):
        self.phase='saving_finalists';self.heartbeat();self.cancel_workers()
        (self.root/'STOP_MATRIX_SERVICE').touch()
        final=[];shortlist=self.shortlist()
        destination=self.archive/'finalists';destination.mkdir(exist_ok=True)
        for row in shortlist:
            name=row['candidate'];source=self.root/'candidates'/name;target=destination/name
            if not source.exists():continue
            files={str(p.relative_to(source)):sha_file(p) for p in source.rglob('*') if p.is_file()}
            if directory_bytes(self.root)+directory_bytes(self.archive)+directory_bytes(source)>60*2**30:
                self.failures.append({'phase':'final_copy','candidate':name,'error':'Copy would exceed storage cap'});continue
            shutil.copytree(source,target,dirs_exist_ok=True)
            for relative,expected in files.items():
                if sha_file(target/relative)!=expected:raise ValueError('Final copy checksum mismatch')
            atomic_json(target/'FILE_HASHES.json',files)
            shutil.rmtree(source)
            row.update(path=str(target),file_hashes=files,mac_validation='pending')
            final.append(row)
        atomic_json(self.root/'FINALISTS.json',final)
        # Retain reproducibility records; queue payloads and export caches are expendable.
        for name in ['logs','evaluation','generation','statistics','search','exports']:
            if (self.root/name).exists():shutil.copytree(self.root/name,self.archive/name,dirs_exist_ok=True)
        for p in self.root.glob('*.json'):
            if p.name!='GPU_ENDPOINT.json':shutil.copy2(p,self.archive/p.name)
        all_rows=self.complete_evaluations();reference=read_json(self.root/'reference_metrics.json',{})
        ids=read_json(self.root/'DEVELOPMENT_SUBSET.json')['ids']
        source_records=[r for r in reference.get('records',[]) if r['id'] in ids]
        scope=read_json(self.root/'SCREENED_RECIPES.json',{})
        status='complete' if len(scope.get('recipes',[]))==24 and not self.failures else 'complete_with_reduced_scope'
        if not final:status='complete_without_verified_finalists'
        summary={'status':status,
            'run_id':self.manifest['run_id'],'finalists':[{k:v for k,v in r.items() if k!='records'} for r in final],
            'complete_evaluations':all_rows,'reference_records':source_records,
            'deferred':'Heldout accuracy and real memory/speed require the next Mac stage',
            'failures':self.failures,'elapsed_seconds':time.time()-self.started}
        atomic_json(self.archive/'RESULTS.json',summary)
        atomic_json(self.root/'RESULTS.json',summary)
        with (self.archive/'results.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['candidate','status','weight_gib','text_cases','image_cases','text_nll','image_nll'])
            for row in all_rows:writer.writerow([row['candidate'],row['status'],row.get('weight_gib'),row['metrics']['text']['cases'],
                row['metrics']['image']['cases'],row['metrics']['text']['nll_per_token'],row['metrics']['image']['nll_per_token']])
        text=['Inkling series 2 completed',f'Run: {self.manifest["run_id"]}',f'Archive: {self.archive}',
            f'Elapsed compute hours: {(time.time()-self.started)/3600:.2f}',
            f'Finalists: {", ".join(r["candidate"] for r in final) or "none"}',
            f'Recorded phase failures: {len(self.failures)}',
            'CPU likelihood comparisons use FP32. Mac heldout quality and memory validation remain separate.',
            'Read RESULTS.json, results.csv and FINALISTS.json for measurements and exact paths.']
        (self.archive/'MORNING_SUMMARY.txt').write_text('\n'.join(text)+'\n')
        shutil.copy2(self.archive/'MORNING_SUMMARY.txt',self.root/'MORNING_SUMMARY.txt')
        self.state.update(status=summary['status'],finished_unix=time.time(),finalists=[r['candidate'] for r in final])
        self.heartbeat();atomic_json(self.archive/'STATUS.json',self.state)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--kit',type=Path,required=True)
    args=p.parse_args();controller=Controller(args.project,args.run,args.kit)
    try:controller.run()
    except BaseException as error:
        controller.failures.append({'phase':controller.phase,'error':repr(error),'traceback':traceback.format_exc()})
        try:controller.finalize()
        except BaseException as final_error:
            controller.state.update(status='failed',error=repr(error),finalization_error=repr(final_error))
            controller.cancel_workers();controller.heartbeat()
        raise

if __name__=='__main__':main()
