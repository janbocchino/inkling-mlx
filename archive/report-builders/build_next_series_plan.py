from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

recipes = []
def recipe(name, method, group, attention=4, embedding=8, protected=None):
    recipes.append(dict(id=name, method=method, group_size=group,
        expert_bits=2, attention_bits=attention, embedding_and_head_bits=embedding,
        expert_layers_at_4_bits=protected or [],
        requires_source_activation_statistics=(method == 'W'),
        status='planned', implemented_by_current_converter=(method == 'R' and attention == 4 and embedding == 8 and not protected)))

for method in ['R', 'C', 'W']:
    for group in [32, 64, 128]:
        recipe(f'{method}{group}', method, group)
for method in ['C', 'W']:
    for group in [32, 64, 128]:
        recipe(f'{method}{group}-A8', method, group, attention=8)
for method in ['C', 'W']:
    for group in [64, 128]:
        recipe(f'{method}{group}-E4', method, group, embedding=4)
for method in ['C', 'W']:
    recipe(f'{method}128-T4', method, 128, protected=[20, 21, 22, 23])
    recipe(f'{method}128-S4', method, 128, protected='top_two_stored_layers_by_calibration_sensitivity')
recipe('W128-COMB', 'W', 128, attention=8, embedding=4,
       protected='top_two_stored_layers_by_calibration_sensitivity')
assert len(recipes) == 24 and len({r['id'] for r in recipes}) == 24

manifest = {
    'title': 'Inkling 2 Mini overnight experiment series',
    'created_local_date': '2026-11-21', 'timezone': 'Europe/Vienna',
    'status': 'plan_only_not_an_executable_runner',
    'goal': 'Smaller or more accurate text and image MLX inference on 16 GiB Apple silicon',
    'source_revision': '1fb4134b9a74f9681054f4ef82e4bf1f458a2337',
    'mlx_lm_revision': '0c4c99192230f1943dd468d297f0b8a64ae7b7c2',
    'mlx_version': '0.32.4', 'seed': 20261121,
    'leonardo': {'account': 'RECORDED_ACCOUNT', 'account_end_date': '2026-11-24',
        'project': '/PROJECT',
        'experiment_scratch_parent': '/SCRATCH/inkling-mlx-series',
        'partition': 'boost_usr_prod', 'qos': 'boost_qos_dbg must not be used',
        'observed_on': '2026-11-21 Europe/Vienna',
        'observed_allocation_total_local_hours': 80000,
        'observed_consumed_local_hours': 4667,
        'observed_work_filesystem_free_reported_G': 151,
        'filesystem_free_is_not_a_user_quota': True},
    'limits': {'series_compute_deadline_hours': 12,
        'cpu_controller': {'cpus': 16, 'memory_GiB': 192, 'walltime_hours': 12, 'gpus': 0},
        'gpu_workers_total': 2, 'gpu_worker': {'cpus': 8, 'memory_GiB': 32,
            'gpus': 1, 'walltime_hours': 3}, 'gpu_hour_ceiling': 6,
        'new_disk_GiB': 60, 'minimum_work_filesystem_free_GiB': 60,
        'saved_finalists_max': 3, 'materialized_new_candidates_max': 6,
        'stop_starting_candidates_at_hour': 10.5, 'finalize_at_hour': 11.5,
        'gpu_queue_wait_limit_minutes': 30,
        'cpu_gpu_math_fallback_if_no_worker': True},
    'methods': {'R': 'Existing affine round to nearest',
        'C': 'Per group clipping chosen by weight reconstruction mean squared error',
        'W': 'Per group clipping chosen by source activation diagonal weighted reconstruction error; not GPTQ or AWQ'},
    'clipping': {'alphas': [1, .99, .98, .97, .95, .925, .90, .85, .80, .75, .70],
        'scope': 'Expert matrices only; other eligible matrices use existing round to nearest at recipe precision',
        'bounds': 'Midpoint of each weight group min and max plus or minus alpha times its half range',
        'cpu_fallback_alphas': [1, .95, .90],
        'final_packing_backend': 'Pinned MLX CPU mx.quantize',
        'gpu_search_requires_packing_parity_pilot': True},
    'calibration': {'target_text_chunks': 96, 'target_images': 24,
        'source_token_budget_including_visual_tokens_max': 24576,
        'minimum_text_chunks': 24, 'minimum_images': 8,
        'max_phase_hours': 2, 'expert_min_routed_tokens_for_weighted_score': 32,
        'rare_expert_fallback': 'C reconstruction objective with explicit coverage report',
        'statistics': 'FP32 sum of squared inputs and counts per projection expert and loop visit; aggregate visits for shared weights',
        'S4_fallback_if_calibration_unavailable': 'Use C weight reconstruction sensitivity times known loop visit count per additional byte'},
    'evaluation': {'development_target_text': 48, 'development_target_images': 24,
        'development_minimum_text': 12, 'development_minimum_images': 6,
        'heldout_text': 120, 'heldout_images': 60,
        'heldout_use': 'Next day Mac comparison after configurations are frozen; full set not promised during CPU overnight run',
        'teacher_forced_answer_token_budget_per_candidate': 8192,
        'generative_smoke_cases_per_finalist_target': 12,
        'generation_max_new_tokens': 192, 'temperature': 0,
        'reasoning_effort': 'none', 'image_long_side_pixels': 512,
        'source_reference': 'Released expert values reconstructed through BF16 then widened to FP32 in host RAM for MLX CPU gather_mm; not recovered training master weights',
        'cpu_comparison_precision': 'FP32 activations and floating parameters for reference and candidates; keep candidate packed integer codes unchanged; Mac confirms deployment behavior',
        'comparison_subset': 'Fix the same stratified prefix for every candidate before scoring using pilot measured throughput'},
    'mac': {'memory_GiB': 16, 'prompt_lengths': [512, 2048, 4096],
        'prefill_steps': [64, 256], 'repeats': 3, 'warmups': 1,
        'preferred_weight_size_GiB': 9.75, 'stretch_weight_size_GiB': 9.2,
        'max_gold_score_loss_vs_current_baseline_percentage_points': 2,
        'max_category_loss_vs_current_baseline_percentage_points': 5,
        'quality_tolerances_are_engineering_filters_not_statistical_claims': True,
        'mac_work_during_night': 'No heavy work required; benchmark finalists the following day'},
    'recipes': recipes,
    'unimplemented_components': ['manifest driven mixed layer precision and audits',
        'in memory FP32 reference and source CPU forward validation',
        'text image calibration hooks with routing and loop visit coverage',
        'CPU and PyTorch CUDA clipping search with parity checks',
        'size and error recipe screening without 24 complete checkpoints',
        'bounded resumable controller plus Slurm worker dispatch',
        'teacher forced and gold answer evaluation and summary report']
}
(OUT / 'Inkling_next_series_matrix.json').write_text(json.dumps(manifest, indent=2) + '\n')

doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin = section.bottom_margin = Inches(.65)
section.left_margin = section.right_margin = Inches(.75)
normal = doc.styles['Normal']
normal.font.name = 'Calibri'
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.paragraph_format.line_spacing = 1.04
normal.paragraph_format.space_after = Pt(7)
for name, size in [('Title', 25), ('Subtitle', 11), ('Heading 1', 17), ('Heading 2', 12)]:
    st = doc.styles[name]
    st.font.name, st.font.size = 'Calibri', Pt(size)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.font.bold = name.startswith('Heading')
    st.paragraph_format.space_before = Pt(10 if name.startswith('Heading') else 0)
    st.paragraph_format.space_after = Pt(7)
    st.paragraph_format.keep_with_next = True
for st in doc.styles:
    pp = st.element.find(qn('w:pPr'))
    if pp is not None:
        for border in list(pp.findall(qn('w:pBdr'))):
            pp.remove(border)
footer = section.footer.paragraphs[0]
footer.alignment = 2
r = footer.add_run('Inkling experiment series  |  ')
r.font.size = Pt(9)
field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)

def p(text):
    return doc.add_paragraph(text)
def h(text):
    doc.add_heading(text, 2)
def page(title):
    doc.add_page_break()
    doc.add_heading(title, 1)
def table(headers, rows, widths, numeric=()):
    t = doc.add_table(rows=1, cols=len(headers)); t.autofit = False
    for col, width in zip(t.columns, widths): col.width = Inches(width)
    for cell, val in zip(t.rows[0].cells, headers): cell.text = str(val)
    for values in rows:
        for cell, value in zip(t.add_row().cells, values): cell.text = str(value)
    borders = OxmlElement('w:tblBorders')
    for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        b = OxmlElement('w:' + edge)
        for k, v in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]: b.set(qn('w:' + k), v)
        borders.append(b)
    t._tbl.tblPr.append(borders)
    for ri, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
        if ri == 0: trpr.append(OxmlElement('w:tblHeader'))
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement('w:tcMar')
            for edge in ['top', 'bottom', 'left', 'right']:
                e = OxmlElement('w:' + edge); e.set(qn('w:w'), '80'); e.set(qn('w:type'), 'dxa'); mar.append(e)
            tcpr.append(mar)
            shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E7EDF2' if ri == 0 else ('F7F9FA' if ri % 2 == 0 else 'FFFFFF')); tcpr.append(shade)
            for para in cell.paragraphs:
                para.paragraph_format.space_before = Pt(1)
                para.paragraph_format.space_after = Pt(1)
                para.paragraph_format.line_spacing = 1.02
                if ci in numeric: para.alignment = 1
                for run in para.runs: run.font.size = Pt(10); run.bold = ri == 0
    blank = doc.add_paragraph(); blank.paragraph_format.space_after = Pt(0); blank.paragraph_format.space_before = Pt(0)
    blank.paragraph_format.line_spacing = .35
    return t
def link(label, url):
    para = doc.add_paragraph()
    el = OxmlElement('w:hyperlink')
    el.set(qn('r:id'), para.part.relate_to(url, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', is_external=True))
    run = OxmlElement('w:r'); pr = OxmlElement('w:rPr'); col = OxmlElement('w:color'); col.set(qn('w:val'), '174A78'); pr.append(col); run.append(pr)
    text = OxmlElement('w:t'); text.text = label; run.append(text); el.append(run); para._p.append(el)
    para.paragraph_format.space_after = Pt(5)
def code(text):
    para = doc.add_paragraph()
    for line in text.splitlines():
        run = para.add_run(line + '\n'); run.font.name = 'Courier New'; run.font.size = Pt(9)

doc.add_paragraph('Inkling 2 Mini overnight experiment plan', 'Title')
doc.add_paragraph('Series 2 for text and image inference on a 16 GB Mac', 'Subtitle')
p('Prepared 21 November 2026, Europe Vienna time. Plan a 12 hour Leonardo run after a supervised pilot. Use the next morning to test the best checkpoints on the Mac.')
h('What we want to learn')
p('Find a checkpoint that uses less memory than our current 2 bit expert model while preserving useful text and image answers. Search 24 recipes, write at most six complete new checkpoints, and retain up to three finalists: the smallest useful model, the best quality model, and a compromise when it is a distinct recipe.')
p('Leonardo will perform calibration, numerical searches, conversion and initial scoring. The Mac will establish whether a finalist is comfortable to use. A fast A100 result cannot establish Mac memory use or latency.')
h('The experiment we already have')
table(['Measure', 'Verified starting point'], [
    ['Weights', '11.02 GB, or 10.26 GiB; expert 2 bit, other eligible linear modules 4 bit, embeddings and output head 8 bit; group size 64'],
    ['Mac result', 'M5 with 16 GiB. Simple text and image checks passed. Short generation was roughly 25 to 29 tokens per second; system swap increased.'],
    ['Conversion cost', 'Successful CPU conversion took 8 min 33 sec and about 15.9 GiB peak host RAM. This is a packing estimate, not a calibration benchmark.'],
    ['Allocation', 'RECORDED_ACCOUNT: 4,667 of 80,000 local hours consumed; account ends 24 November 2026. Checked by SSH on 21 November.'],
    ['Storage', 'Work filesystem reported about 151 GiB free and the project occupied about 30 GiB. Filesystem free space is not a personal quota.']
], [1.25, 5.75])
h('What success would look like')
p('Prefer weights at or below 9.75 GiB, with 9.2 GiB as a stretch target. A candidate should lose no more than 2 percentage points overall against the current baseline on the final task set, with no category dropping more than 5 points. These are engineering filters; the small sample cannot prove statistical equivalence. Report the gap from the released source reference separately.')
p('The main morning deliverable is a ranked results table and verified model folders. A recipe can be smaller without being good enough to deploy. If every smaller recipe fails, retain the current model and report the measured tradeoff.')

page('The 24 recipes to compare')
p('A group shares quantization scales and offsets. Larger groups reduce metadata but can introduce more error. Clipping slightly narrows the weight range before choosing the four codes available at 2 bits. The search tests whether that tradeoff preserves useful behavior.')
table(['Method', 'How it chooses the range'], [
    ['R', 'Existing round to nearest conversion; no clipping search. R64 is the current baseline.'],
    ['C', 'Choose clipping to minimize weight reconstruction error, measured over each group.'],
    ['W', 'Choose clipping using reconstruction error weighted by observed source activations. Common inputs count more. This is a diagonal approximation, not GPTQ or AWQ.']
], [0.75, 6.25])
table(['Recipe family', 'Variants', 'Count'], [
    ['R, C and W', 'Group sizes 32, 64 and 128', 9],
    ['C and W with A8', 'Each group size; attention linears at 8 bits', 6],
    ['C and W with E4', 'Groups 64 and 128; embeddings and output head at 4 bits', 4],
    ['C and W with T4', 'Group 128; expert stacks in stored layers 20 to 23 at 4 bits', 2],
    ['C and W with S4', 'Group 128; expert stacks in the two most sensitive stored layers at 4 bits', 2],
    ['W128 COMB', 'Combine A8, E4 and S4 after the individual comparisons', 1],
    ['Total', 'Includes the existing R64 baseline', 24]
], [1.45, 4.9, .65], numeric=(2,))
p('All recipes start from the original release. Clip expert matrices only; use existing round to nearest for the other matrices at the recipe precision. Preserve the vision encoder, router, loop controllers, norms, convolutions, all experts and all loop passes. A8 changes attention linears only; other eligible non expert linears stay at 4 bits. Nondivisible dimensions retain the established source precision rule.')
p('Select S4 layers by calibration reconstruction sensitivity per additional byte, aggregated over loop visits. Without calibration, use C weight error times the known visit count. Use the same selected layers for C and W. Protect whole expert stacks, because a packed SwitchLinear module has one bit width. Record exact layer numbers before development scoring.')
h('Search the numbers before writing the files')
p('Try clipping factors 1.00, 0.99, 0.98, 0.97, 0.95, 0.925, 0.90, 0.85, 0.80, 0.75 and 0.70 around each group midpoint. Keep alpha 1.00 as the no clipping option. Cache per tensor costs and exact predicted bytes, then assemble recipe estimates. This avoids writing 24 full models. Weight error is a screening signal; final selection requires model outputs.')

page('How Leonardo will do the work')
h('A reference that can run on the available hardware')
p('The released expert weights are NVFP4 and MXFP8. Reconstruct their represented values through BF16, widen to FP32, and replace native packed experts with ordinary floating point expert modules. MLX CPU gather_mm requires FP32. A tiny local CPU text test passed in FP32; BF16 failed. Use FP32 floating parameters and activations for both reference and candidate CPU comparisons, leaving candidate packed integer codes unchanged.')
p('This reference preserves the released represented weight values, not higher precision training weights. Its CPU arithmetic differs from Mac deployment arithmetic, so the Mac remains the final check. Validate tiny image and full source forwards before calibration. Keep roughly 106 GB of reference weights in host RAM within the 192 GiB job request; never save this full reference checkpoint to disk. Release it before candidate inference.')
h('Use the A100s for the numerical search')
p('Two independent PyTorch CUDA workers can each score batches of reconstructed weight tiles and input statistics. Each worker processes disjoint layers and returns clipping choices and error summaries. A complete Inkling PyTorch port is not required for this route. Final MLX packing and strict reload remain in the known CPU environment.')
p('Use the already tested CUDA 12.1 PyTorch environment if its required tensor operations pass the pilot. The last observed driver was 535.274.02. Current MLX CUDA wheels require a newer driver, and a container alone does not update the host driver. Do not build the plan around native MLX CUDA or hardware NVFP4 inference on the A100s. See sources 1 to 3.')
table(['Job', 'Resource request', 'Limit'], [
    ['CPU controller', '16 CPU cores, 192 GiB RAM, zero GPUs; boost_usr_prod normal QoS', '12 hours'],
    ['GPU workers', 'Up to 2 workers, each 1 A100, 8 CPU cores and 32 GiB host RAM', '3 hours each'],
    ['New artifacts', 'Up to 6 full new exports over the run; retain up to 3 finalists', '60 GiB total new disk use'],
    ['Concurrent work', 'No more than one full reference or candidate inference process', 'Other account jobs remain untouched']
], [1.35, 4.6, 1.05])
h('Pilot requirements before the unattended run')
p('Check finite logits and text plus image forwards on the reconstructed reference. Measure actual reference tokens per second, expert routing coverage, one candidate export time and GPU tile throughput. Compare the CUDA scorer against the final pinned MLX quantize and dequantize path on representative tiles, including zero ranges, constants and extreme values. Require matching reconstructed values within a declared dtype tolerance; do not use a scorer with unresolved packing or rounding differences.')
p('If CUDA scoring fails or workers cannot start within 30 minutes, use CPU scoring with factors 1.00, 0.95 and 0.90. If the full source calibration cannot meet its minimum within two hours, skip the 11 W recipes and run the 13 R and C recipes. Mark the reduced scope in the results. Do not substitute activations from our 2 bit baseline and call them source calibration.')

page('The overnight schedule')
p('Elapsed times start when the CPU controller gets its allocation. Queue time before that is extra. The controller enforces one 12 hour deadline across all workers, stops launching new candidates at 10.5 hours and begins finalization at 11.5 hours. Durations below are budgets to validate with the pilot, not measured forecasts.')
table(['Elapsed time', 'Work', 'Checkpoint produced'], [
    ['0 to 0.5 h', 'Validate source, environment, manifests, dataset hashes, disk space and reference path. Run bounded pilots.', 'Preflight report and measured time estimates'],
    ['0.5 to 2.5 h', 'Collect source activation statistics and reference answer likelihoods. Prioritize a balanced minimum text and image sample.', 'Reference metrics, routing counts and calibration statistics'],
    ['2.5 to 5.5 h', 'GPU tile search, exact recipe size estimates, sensitivity ranking and screening of 24 recipes.', 'Per tensor search cache and ranked recipe estimates'],
    ['5.5 to 10.5 h', 'Export and score up to 6 candidates sequentially, including strict reload and hashes. Keep a rolling shortlist of at most 3 complete models.', 'Verified folders, development scores and selection record'],
    ['10.5 to 11.5 h', 'Run bounded generative text and image checks on finalists; finish any in progress comparison.', 'Example answers and failure categories'],
    ['11.5 to 12 h', 'Flush metrics, verify manifests, retain finalists and write the morning summary.', 'Results table, hashes and transfer instructions']
], [1.05, 3.8, 2.15])
h('A bounded adaptive search')
p('Always include R64 in development scoring. Screen by exact bytes and reconstruction error within each method; avoid treating C and W objective values as directly comparable scores. Promote R128, the smallest promising recipe, the best estimated quality recipe, and representatives of clipping, activation weighting and layer protection, up to six new exports. A candidate dominated in both bytes and the same error objective can be pruned early.')
p('Use pilot throughput to fix a stratified development subset that every candidate can finish in the available time. Target all 72 cases; the minimum useful overnight comparison is 12 text and 6 image cases. Record the subset before scoring the first candidate. If even the minimum cannot fit, publish conversion results without naming an accuracy winner. Do not allow faster candidates to receive easier or fewer cases.')
p('Score each export immediately, then remove its files if it falls outside the rolling shortlist. Keep its recipe, hashes and scores so it can be reproduced. Check predicted disk use before every export; reduce the shortlist if three retained models plus the next export would exceed 60 GiB.')
p('The CPU controller may submit the two GPU workers when statistics are ready. Workers read a bounded disk queue and exit when it is exhausted. Pending workers are cancelled when their wait budget or the series deadline expires. If the controller exits or fails, workers stop on its expired heartbeat. Successful batch jobs continue after the SSH session closes.')

page('Calibration and quality evaluation')
h('Separate examples used for tuning from examples used for judging')
table(['Split', 'Target size', 'Purpose'], [
    ['Calibration', '96 text chunks and 24 images; at most 24,576 combined text and visual tokens', 'Collect statistics and choose clipping. Minimum 24 text chunks and 8 images.'],
    ['Development', '48 text questions and 24 image questions', 'Rank the exported recipes. Use the same fixed subset if the CPU budget is tight.'],
    ['Held out', '120 text questions and 60 image questions', 'Next day comparison of the frozen finalists and current baseline on the Mac.']
], [1.0, 2.55, 3.45])
p('Freeze IDs, dataset revisions, random seed 20261121, answer keys, scoring rules and image hashes before the search. Split by source article, document, image and generated task template so near duplicates cannot cross splits. Keep held out answers out of calibration, clipping and layer selection.')
p('Use WikiText training passages plus authored extraction, arithmetic, instruction and short code inputs for text calibration. Include real chart and scene text images from official ChartQA and TextVQA releases alongside generated layout tests. Use separate source splits or disjoint documents and images for development and final evaluation. Record the applicable data terms and citations in the manifest. Download small pinned subsets before submission so jobs can run offline.')
h('Collect statistics without storing huge activation dumps')
p('For every expert projection, accumulate FP32 sums of squared input coordinates and routed token counts. Track each loop visit, then combine visits for the shared stored weight. Include image conditioned inputs. An expert with fewer than 32 routed calibration tokens uses the C objective; report its fallback explicitly. Save compact statistics, not full Hessians or all hidden states. Keep the total new artifact budget at 60 GiB.')
h('Score behavior as well as numerical error')
p('For every exported candidate and R64, measure answer token negative log likelihood against the same gold continuations, and report its increase over the reconstructed source reference. Mask prompt and image tokens from the loss. Cap scored answer tokens at 8,192 per model. Keep text and image results separate. This provides a repeatable teacher forced comparison without generating a long answer for every recipe.')
p('Run up to 12 generative smoke cases per finalist during the final hour. Use greedy decoding, the same chat template, reasoning effort none, a 512 pixel longest image side and a maximum of 192 new tokens, including reasoning. A response that exhausts the budget without a usable final answer is a failure. Short tests must include OCR, chart reading and spatial relations as well as text and code.')
p('On the held out set, score extraction and arithmetic by exact answer, structured output by parsing, and code by inspected isolated tests. Use published dataset answer normalization for chart and scene text questions. Keep free descriptions as a separate human review category. Report denominators, per category accuracy, uncertainty intervals and paired changes. The full 180 case Mac evaluation is a next day task, not a promised CPU overnight result.')

page('Recovery storage and reproducibility')
h('Keep useful results even when a phase fails')
p('Use a new experiment ID such as series2 followed by a timestamp. Write progress and one JSON record per completed tensor, recipe or case. Each record contains the source hash, recipe hash, runtime version, seed and dataset manifest hash. On resume, reuse only records whose inputs match. The current converter does not supply this series controller or partial tensor resume; these must be implemented before launch.')
p('Export to a candidate specific incomplete directory and publish its final name only after strict reload succeeds. Checkpoint at tensor boundaries and handle termination signals. The Slurm time warning starts finalization; the controller deadline provides an earlier fallback. Retry a failed tensor once. After a repeated error, mark the affected candidate failed and continue with valid work. Never turn a missing result into a zero error score.')
h('Use scratch for temporary search files')
p('Place temporary tiles, caches and incomplete exports below /SCRATCH/inkling-mlx-series. Confirm the actual user and project quota before starting; the large scratch filesystem free space does not establish the quota. Copy only compact reports and verified finalists back into the project. Scratch is temporary storage, so the morning transfer is part of completing the run.')
p('Cap all new files across work and scratch at 60 GiB, with no more than three retained full finalists and one candidate being exported at a time. Keep at least 60 GiB free on the work filesystem. Use a bounded tile queue rather than writing the full FP32 reference. Account for temporary duplication when copying finalists back to work. Remove only expendable intermediates listed in this experiment manifest. Preserve the source, validated R64 checkpoint and other projects.')
h('Save enough evidence to reproduce the ranking')
table(['Output', 'What it must contain'], [
    ['RUN_MANIFEST.json', 'Source and runtime revisions, recipes, data IDs, resource limits, pilot measurements and actual scope'],
    ['calibration and search records', 'Loop and expert coverage, fallback decisions, per tensor costs and clipping parameters'],
    ['results.csv and results.json', 'Actual bytes, precision map, export checks, text and image likelihoods, gold scores and incomplete cases'],
    ['finalists and hashes', 'Up to three self contained MLX folders with assets, strict reload results and file checksums'],
    ['MORNING_SUMMARY.txt', 'What finished, what failed, remaining work, chosen finalists, transfer commands and measured resource use']
], [1.8, 5.2])
p('Record Slurm job IDs, elapsed time, allocated resources and MaxRSS from accounting. At most two 3 hour single GPU workers means a 6 GPU hour request ceiling. CINECA allocation local hours follow its accounting rules; GPU hours are not interchangeable with the saldo balance. Recheck the account before submission because it expires on 24 November.')

page('How to turn this plan into an overnight run')
h('Prepare on the Mac and submit on Leonardo')
p('Use the Mac project to build the series runner, keeping the working launcher and baseline unchanged. Upload the new runner and manifests to the existing Leonardo inkling mlx project. Submit through Slurm from the Leonardo terminal; conversion and scoring must run on compute nodes. Once the pilot passes, one submission should start the controller and its bounded worker jobs.')
table(['Build step', 'Required result before launch'], [
    ['1 Extend the converter', 'Accept a per module recipe, clipping choices and mixed layer precision; make audits validate that exact recipe. Existing code supports only R with group 32, 64 or 128.'],
    ['2 Add the reference and collectors', 'Reconstruct expert values through BF16 and widen to FP32 in RAM for CPU forwards. Gather per expert and loop statistics for text and images.'],
    ['3 Add the search and scorer', 'Implement C and W, the CUDA parity pilot, exact size estimates, development scoring and a results table.'],
    ['4 Add the controller', 'Implement time and storage limits, checkpoints, worker submission and cleanup, resume, and the morning summary. Use normal boost_usr_prod QoS, not the 30 minute debug QoS.'],
    ['5 Run the supervised pilot', 'Test one real layer search, one source text and image example, one candidate export and resume after an interruption. Use measurements to set the overnight scope.']
], [1.65, 5.35])
p('The companion JSON specifies all 24 recipe IDs and their limits. It is a planning manifest, not a runnable command. The existing conversion kit cannot yet run this series. The next implementation should supply a single launcher plus status and resume commands, then validate them during the pilot.')
h('The following morning')
p('Read the summary before transferring any weights. Download at most two distinct finalists initially, verify their hashes, then compare them with R64 on the 16 GiB Mac. Run one model at a time with other large apps closed. Use 512, 2048 and 4096 token prompts, prefill steps 64 and 256, one warmup and three measured repeats. Separate cold load time, time to first token and output generation speed.')
p('Record MLX peak memory, process RSS, system memory pressure and swap before and after each run. Compare swap deltas under similar background load; existing swap may not disappear between runs. For the final choice, require text and image quality to pass the predefined filters and memory pressure to improve or remain acceptable without sustained growth during repeated runs. A lower file size alone does not establish a comfortable fit.')
h('Sources and implementation references')
link('1 CINECA Leonardo hardware partitions and job limits', 'https://docs.hpc.cineca.it/hpc/leonardo.html')
link('2 MLX installation requirements for CPU and CUDA', 'https://ml-explore.github.io/mlx/build/html/install.html')
link('3 NVIDIA CUDA minor version compatibility and limitations', 'https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html')
link('4 Pinned MLX LM GPTQ implementation and supported module handling', 'https://github.com/ml-explore/mlx-lm/blob/0c4c99192230f1943dd468d297f0b8a64ae7b7c2/mlx_lm/quant/gptq.py')
link('5 Salesforce WikiText dataset', 'https://huggingface.co/datasets/Salesforce/wikitext')
link('6 Official ChartQA repository and dataset', 'https://github.com/vis-nlp/ChartQA')
link('7 Official TextVQA project', 'https://textvqa.org/')

doc.core_properties.title = 'Inkling 2 Mini overnight experiment plan'
doc.core_properties.subject = 'Series 2 quantization search on Leonardo for 16 GiB Apple silicon'
doc.core_properties.author = 'Inkling experiment project'
doc.save(OUT / 'Inkling_next_experiment_plan.docx')
print(json.dumps({'recipes': len(recipes), 'weighted_recipes': sum(x['method'] == 'W' for x in recipes), 'docx': str(OUT / 'Inkling_next_experiment_plan.docx')}))
