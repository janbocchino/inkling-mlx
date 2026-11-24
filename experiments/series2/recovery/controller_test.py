import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path('outputs/series2-kit').resolve()))
from controller import Controller
from common import atomic_json
with tempfile.TemporaryDirectory() as d:
 p=Path(d);root=p/'run';root.mkdir();project=p/'project';project.mkdir();kit=p/'kit';kit.mkdir()
 atomic_json(root/'RUN_MANIFEST.json',{'run_id':'test'})
 atomic_json(root/'DEVELOPMENT_SUBSET.json',{'ids':['a','b']})
 (root/'evaluation').mkdir();(root/'candidates').mkdir()
 for name,size,nll in [('C128',100,2.),('W128',110,1.8),('C64',120,2.2),('C32',130,2.3)]:
  folder=root/'candidates'/name;folder.mkdir();(folder/'owned').write_text('x')
  atomic_json(folder/'CONVERSION_REPORT.json',{'weight_bytes':size})
  atomic_json(root/'evaluation'/(name+'.json'),{'status':'complete','candidate':name,'records':[{'id':'a'},{'id':'b'}],
   'metrics':{'text':{'nll_per_token':nll},'image':{'nll_per_token':nll}}})
 preserved=project/'source';preserved.mkdir();(preserved/'untouched').write_text('source')
 c=Controller(project,root,kit)
 chosen=c.shortlist();assert len(chosen)==3
 assert {'C128','W128'}.issubset({r['candidate'] for r in chosen})
 c.prune();assert (preserved/'untouched').read_text()=='source'
 assert len(list((root/'candidates').iterdir()))==3
 print('Controller shortlist and cleanup tests passed')
