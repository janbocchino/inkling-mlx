"""Freeze small text/image datasets locally before the offline cluster run."""
from __future__ import annotations
import io, json, random, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import atomic_json, sha_file, digest, event

SEED = 20261121
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

def get_json(url):
    return json.load(urllib.request.urlopen(url, timeout=60))

def public_rows(dataset, split, count, columns=None):
    import fsspec
    import pyarrow.parquet as pq
    meta = get_json('https://huggingface.co/api/datasets/' + dataset)
    revision = meta['sha']
    names = [s['rfilename'] for s in meta['siblings'] if s['rfilename'].startswith('data/' + split + '-') and s['rfilename'].endswith('.parquet')]
    if not names: raise ValueError(f'No parquet files for {dataset} {split}')
    url = f'https://huggingface.co/datasets/{dataset}/resolve/{revision}/{names[0]}'
    result = []
    with fsspec.open(url, 'rb', block_size=2**20) as f:
        parquet = pq.ParquetFile(f)
        for batch in parquet.iter_batches(batch_size=64, columns=columns):
            result.extend(batch.to_pylist())
            if len(result) >= count: break
    return result[:count], {'dataset': dataset, 'revision':revision, 'split':split, 'file':names[0]}

def save_image(image, name):
    image = image.convert('RGB'); image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    filename = f'images/{name}.npy'
    path = DATA / filename; path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(image, dtype=np.uint8), allow_pickle=False)
    return filename

def authored_text(split, count):
    rng = random.Random(SEED + {'calibration':1, 'development':2, 'heldout':3}[split])
    result = []
    for i in range(count):
        category = ['arithmetic', 'extraction', 'structured', 'logic', 'code', 'instruction'][i % 6]
        a,b,c = [rng.randint(5, 89) for _ in range(3)]
        if category == 'arithmetic':
            q = f'Compute {a} times {b}, then add {c}. Reply with only the integer.'; ans = str(a*b+c)
        elif category == 'extraction':
            order = f'{split[0].upper()}{rng.randrange(100000,999999)}'
            q = f'Extract only the shipment code. Note: order {order} contains {a} books; its warehouse code is WH-{b}.'; ans=order
        elif category == 'structured':
            q=f'Return valid JSON only, with keys "name" and "count". Name: item_{split}_{i}. Count: {a}.'
            ans=json.dumps({'name':f'item_{split}_{i}', 'count':a}, separators=(',',':'))
        elif category == 'logic':
            q=f'A box starts with {a} red balls and {b} blue balls. We add {c} blue balls. How many balls are now in the box? Reply with only the integer.'; ans=str(a+b+c)
        elif category == 'code':
            q=f'In Python, what is the value of sum(x * 2 for x in [{a}, {b}, {c}])? Reply with only the integer.'; ans=str(2*(a+b+c))
        else:
            bank={'calibration':['cedar','violet','amber','harbor','cobalt','meadow'],
                'development':['maple','scarlet','ivory','orchard','indigo','summit'],
                'heldout':['willow','crimson','silver','lagoon','ochre','prairie']}[split]
            word=bank[(i//6)%6]+'_'+str(i)
            q=f'Return exactly this string in uppercase, with no explanation: {word}'; ans=word.upper()
        result.append({'id': f'{split}-text-{i:03d}', 'question':q, 'answer':ans,
            'answers':[ans], 'category':category, 'modality':'text', 'origin':'authored',
            'template_group':category})
    return result

def generated_images(split, count):
    # Templates differ between splits as well as values; these are diagnostic tasks.
    rng=random.Random(SEED+{'calibration':11,'development':22,'heldout':33}[split])
    result=[]
    try: font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
    except OSError: font=ImageFont.load_default(size=24)
    for i in range(count):
        image=Image.new('RGB',(360,240),'white'); draw=ImageDraw.Draw(image)
        if i%2==0:
            color = ['red','blue','green','orange'][rng.randrange(4)]
            if split=='calibration':
                draw.rectangle((20+i,60,120+i,160),fill=color);draw.ellipse((220,60,320,160),fill='black')
                q='What color is the rectangle on the left? Reply with one color word.'
            elif split=='development':
                draw.ellipse((20,60,120,160),fill='black');draw.rectangle((220-i,60,320-i,160),fill=color)
                q='What color is the rectangle on the right? Reply with one color word.'
            else:
                draw.rectangle((20,20,120,90),fill='black');draw.ellipse((180+2*i,110,280+2*i,210),fill=color)
                q='What color is the lower circle? Reply with one color word.'
            answer=color;category='spatial'
        else:
            number=str(rng.randint(1000,9999))
            draw.text((30,70),f'{split[:3].upper()}{i} {number}',fill='black',font=font)
            draw.text((30,130),'Read the four digits',fill='black',font=font)
            q='Read the four digit number in this image. Reply with only the digits.'
            answer=number;category='ocr_generated'
        filename=save_image(image,f'{split}-generated-{i:03d}')
        result.append({'id':f'{split}-generated-{i:03d}','question':q,'answer':answer,'answers':[answer],
            'category':category,'modality':'image','image':filename,'origin':'generated',
            'template_group':f'{split}-{category}'})
    return result

def main():
    DATA.mkdir(parents=True,exist_ok=True)
    sources=[]; splits={k:[] for k in ['calibration','development','heldout']}
    for split,count in [('calibration',96),('development',48),('heldout',120)]:
        splits[split].extend(authored_text(split,count))
    # Use genuinely different WikiText articles for raw calibration passages only.
    wiki=get_json('https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1&split=train&offset=0&length=100')
    passages=[r['row']['text'] for r in wiki['rows'] if len(r['row']['text'])>250][:48]
    for i,text in enumerate(passages):
        splits['calibration'][i]['question']=text[:800]
        splits['calibration'][i]['answer']=''
        splits['calibration'][i]['origin']='Salesforce/wikitext training passage'
        splits['calibration'][i]['category']='prose'
    sources.append({'dataset':'Salesforce/wikitext','split':'train','retrieval':'frozen viewer rows 0 to 99',
        'observed_revision':get_json('https://huggingface.co/api/datasets/Salesforce/wikitext')['sha']})
    # The evaluation mirrors provide compact, revision-addressed parquet shards.
    chart,meta=public_rows('lmms-lab/ChartQA','test',180,['type','question','answer','image'])
    sources.append(meta); seen=set(); usable=[]
    for row in chart:
        raw=row['image']['bytes']; key=__import__('hashlib').sha256(raw).hexdigest()
        if key in seen: continue
        seen.add(key);usable.append((row,key))
        if len(usable)==28:break
    if len(usable)<28:raise ValueError('Need 28 distinct chart images')
    for i,(row,key) in enumerate(usable):
        split='development' if i<8 else 'heldout'
        filename=save_image(Image.open(io.BytesIO(row['image']['bytes'])),f'{split}-chart-{i}')
        ans=str(row['answer'])
        splits[split].append({'id':f'{split}-chart-{i}','question':row['question']+' Give a short answer.',
            'answer':ans,'answers':[ans], 'category':'chart','modality':'image','image':filename,
            'origin':'lmms-lab/ChartQA','source_image_sha256':key})
    # TextVQA train for calibration; disjoint validation image IDs for scoring.
    for source_split,needed in [('train',8),('validation',28)]:
        rows,meta=public_rows('lmms-lab/textvqa',source_split,100,
            ['image_id','question','answers','image'])
        sources.append(meta); chosen=0
        for row in rows:
            key='textvqa-'+row['image_id']
            if key in seen or not row['answers']:continue
            seen.add(key)
            split='calibration' if source_split=='train' else ('development' if chosen<8 else 'heldout')
            ans=Counter(row['answers']).most_common(1)[0][0]
            filename=save_image(Image.open(io.BytesIO(row['image']['bytes'])),f'{split}-{key}')
            splits[split].append({'id':f'{split}-{key}','question':row['question']+' Give a short answer.',
                'answer':ans,'answers':row['answers'],'category':'scene_text','modality':'image',
                'image':filename,'origin':'lmms-lab/textvqa','source_image_id':row['image_id']})
            chosen+=1
            if chosen==needed:break
        if chosen<needed:raise ValueError('Not enough disjoint TextVQA images')
    for split,count in [('calibration',16),('development',8),('heldout',20)]:
        splits[split].extend(generated_images(split,count))
    # Development is interleaved before any timing based truncation.
    for split,rows in splits.items():
        text=[r for r in rows if r['modality']=='text']; images=[r for r in rows if r['modality']=='image']
        random.Random(SEED+71).shuffle(text); random.Random(SEED+72).shuffle(images)
        ordered=[]
        while text or images:
            for _ in range(2):
                if text:ordered.append(text.pop())
            if images:ordered.append(images.pop())
        atomic_json(DATA/(split+'.json'),ordered)
    files={str(p.relative_to(DATA)):sha_file(p) for p in DATA.rglob('*') if p.is_file() and p.name!='DATA_MANIFEST.json'}
    manifest={'seed':SEED,'sources':sources,'files':files,
        'counts':{s:dict(Counter(r['modality'] for r in rows)) for s,rows in splits.items()},
        'scope':'Custom frozen diagnostic subsets; not official full benchmark scores',
        'deviations_from_plan':['ChartQA test images are used only for disjoint development and heldout subsets, never calibration.',
            'Authored text task categories share broad operations across splits; values and image templates are distinct. This is a diagnostic suite, not a generalization benchmark.'],
        'data_terms_sources':['https://github.com/vis-nlp/ChartQA','https://textvqa.org/',
            'https://huggingface.co/datasets/Salesforce/wikitext']}
    manifest['content_hash']=digest(manifest)
    atomic_json(DATA/'DATA_MANIFEST.json',manifest)
    event('dataset_frozen',**manifest['counts'],hash=manifest['content_hash'])

if __name__=='__main__':main()
