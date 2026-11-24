from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
conversion = json.loads((OUT / "CONVERSION_REPORT.json").read_text())
probe_path = OUT / "MAC_INFERENCE_RESULTS.json"
probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
context_path = OUT / "MAC_CONTEXT_RESULTS.json"
context = json.loads(context_path.read_text()) if context_path.exists() else {}
tuning_path = OUT / "MAC_TUNING_RESULTS.json"
tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
download_path = OUT / "DOWNLOAD_VERIFICATION.json"
download = json.loads(download_path.read_text()) if download_path.exists() else {}

doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin = section.bottom_margin = Inches(.68)
section.left_margin = section.right_margin = Inches(.75)
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor(0, 0, 0)
normal.paragraph_format.line_spacing = 1.06
normal.paragraph_format.space_after = Pt(7)
for name, size in [("Title", 25), ("Subtitle", 11), ("Heading 1", 17), ("Heading 2", 12)]:
    style = doc.styles[name]
    style.font.name = "Calibri"
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.bold = name.startswith("Heading")
    style.paragraph_format.space_before = Pt(10 if name.startswith("Heading") else 0)
    style.paragraph_format.space_after = Pt(7)
    style.paragraph_format.keep_with_next = True
footer = section.footer.paragraphs[0]
footer.alignment = 2
run = footer.add_run("Inkling experiment  |  ")
run.font.size = Pt(9)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def p(text, bold_lead=None):
    paragraph = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        paragraph.add_run(bold_lead).bold = True
        paragraph.add_run(text[len(bold_lead):])
    else:
        paragraph.add_run(text)
    return paragraph


def h(text):
    doc.add_heading(text, 2)


def page(title):
    if title == "Understanding the model and the compression":
        doc.add_page_break()
    doc.add_heading(title, 1)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    if widths:
        for column, width in zip(t.columns, widths):
            column.width = Inches(width)
    for i, value in enumerate(headers):
        t.rows[0].cells[i].text = value
    for values in rows:
        for cell, value in zip(t.add_row().cells, values):
            cell.text = str(value)
    for index, row in enumerate(t.rows):
        for i, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[i])
            tcpr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for side in ("top", "bottom", "left", "right"):
                element = OxmlElement("w:" + side)
                element.set(qn("w:w"), "75")
                element.set(qn("w:type"), "dxa")
                margins.append(element)
            tcpr.append(margins)
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(3)
                para.paragraph_format.space_before = Pt(3)
                for r in para.runs:
                    r.font.size = Pt(10)
                    if index == 0:
                        r.bold = True
            if index == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "EEEEEE")
                tcpr.append(shade)
        trpr = row._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        trpr.append(no_split)
        if index == 0:
            trpr.append(OxmlElement("w:tblHeader"))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def code(text):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(7)
    for line in text.splitlines():
        r = paragraph.add_run(line + "\n")
        r.font.name = "Courier New"
        r.font.size = Pt(9)
    return paragraph


def link(label, url):
    paragraph = doc.add_paragraph()
    hyperlink = OxmlElement("w:hyperlink")
    rid = paragraph.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "174A78")
    properties.append(color)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    paragraph.paragraph_format.space_after = Pt(5)


doc.add_paragraph("Inkling 2 Mini quantization experiment", "Title")
doc.add_paragraph("What we did and what it means for a 16 GB Mac", "Subtitle")
p("Experiment started 20 November 2026. Overnight follow-up 21 November 2026. Dates use Europe Vienna time. Prepared by Codex for this project.")
h("The result so far")
p("We created an MLX checkpoint with 2-bit expert weights on CINECA Leonardo and downloaded it to the 16 GB M5 Mac. Its weights occupy 11.02 GB, or 10.26 GiB. Every saved and downloaded tensor passed verification. The model then answered four simple text and image probes correctly and retrieved a supplied identifier in prompts up to about 4096 tokens. This is a successful first experiment, with a memory constraint: system swap use increased substantially during inference.")
p("GB counts bytes in decimal units; GiB counts them in powers of 1024. Both size figures describe the same files.")
table(["Question", "Recorded result"], [
    ("What was the goal", "Text and image inference on cheaper Apple silicon Macs with 16 GB of memory"),
    ("Which model", "Thinking Machines Inkling 2 Mini, about 26.4 billion total parameters"),
    ("What was converted", "The original released NVFP4 and MXFP8 checkpoint"),
    ("How much smaller", "About 11.02 GB of output weights versus roughly 19.5 GB of original tensor data"),
    ("Did the conversion finish", "Yes. Job 64991140 completed with exit code 0 in 8 minutes 33 seconds"),
    ("Were files checked", "Yes. Strict reload plus 1567 matching tensor signatures"),
    ("Is quality proven", "No. Structural checks and small smoke tests are different from accuracy benchmarks"),
    ("Did it run on the Mac", "Yes. Short output generation was about 25 to 29 tokens per second; memory pressure and longer-prompt latency need attention"),
], [1.8, 5.2])
h("The experiment in one sentence")
p("We reduced the storage precision of the large expert matrices while retaining higher precision for parts that may be more sensitive, then preserved the architecture, image encoder and tokenizer so the result can be tested as the same model.")
p("This report covers the planning, access problems, environment checks, implementation, conversion, local download and follow-up tests. It records corrections made during the chat as well as the successful steps.")

page("Understanding the model and the compression")
h("Parameters and bits")
p("A parameter is a learned number in the model. Quantization stores many of these numbers with fewer bits. Two bits provide four possible codes within each group. The group also stores a scale and offset that map those codes back to useful numerical values. Compression introduces error; the practical question is how much useful behavior survives.")
p("A simple calculation would put 26.4 billion numbers at two bits each at 6.6 GB. Our result is larger because group metadata also uses space and several components deliberately use more than two bits. A label such as 2-bit therefore does not describe every byte of this checkpoint.")
h("Experts and loops")
p("Inkling 2 Mini is a mixture of experts model. A token uses only a selection of its expert networks, reducing computation compared with using every expert. The usual local setup still stores all expert weights, so the total parameter count matters for Mac memory even when the active computation is smaller.")
p("The model has 24 stored decoder layers. Layers 0 to 5 run once, layers 6 to 17 run three times using shared weights, and layers 18 to 23 run once. That makes 48 layer applications per token without storing 48 separate layers. The repeated passes need separate attention and convolution state. We preserved this schedule.")
h("The terms used in the chat")
table(["Term", "Plain meaning"], [
    ("MLX", "Apple's array framework. MLX-LM supplies model loaders and text generation tools built on it"),
    ("Checkpoint", "The weights and configuration files needed to load a model"),
    ("BF16", "A 16-bit floating point format used for many original non-expert weights"),
    ("NVFP4 and MXFP8", "The low-precision floating point formats used in the released expert weights"),
    ("Affine quantization", "Integer codes plus a scale and offset for each group of numbers"),
    ("KV cache", "Memory used to retain attention information as a conversation grows"),
    ("Calibration", "Using representative examples to choose compression settings that better preserve behavior"),
    ("GPTQ and AWQ", "More sophisticated approaches to post-training quantization; not used in this first conversion"),
], [1.65, 5.35])
p("The official checkpoint was already quantized. Reconstructing its released numbers in BF16 lets us repack them, but it does not recover higher-precision training weights. Model architecture details come from the original configuration and the Mini port listed in the sources.")

page("How we got Leonardo ready")
h("Clarifying the target")
p("You started with no model quantization experience, access to a sizeable Leonardo allocation, and a goal of cheaper Mac inference. You selected 16 GB Apple silicon and text plus images. You later clarified that you know basic coding. The workflow shifted to model-specific guidance rather than basic terminal instruction.")
p("The Mac reported arm64, macOS 26.6.2 and Python 3.13.13. Leonardo is much more powerful for large experiments, while the Mac is essential for testing the actual deployment constraints. For this first conversion we needed CPU processing and substantial memory rather than training on multiple GPUs.")
h("Getting connected")
p("Two SSH failures appeared. The leonardo shortcut reached a login node but rejected authentication. The generic login address produced a changed host identity warning. We explained the separate roles of SSH connection settings and the known_hosts server identity cache.")
p("CINECA uses temporary SSH authentication certificates, normally valid for 12 hours. The suggested renewal command used the CINECA registered email and browser OTP login. Its documentation also explains why a generic cluster address can point to different login nodes. The guidance was to confirm a changed server fingerprint before replacing its saved identity. The chat records that you subsequently logged in; it does not record exactly which local SSH changes you made. No claimed verification of the RSA fingerprint should be inferred.")
h("Choosing the active allocation")
p("The account HISTORICAL_ACCOUNT had ended on 27 July 2026. We used RECORDED_ACCOUNT, active from 24 August through 24 November 2026. The reported allocation was 80000 hours with 4662 consumed. The work area was /LEONARDO_WORK; the user scratch area was also available. A personal project folder was created inside the shared work area.")
h("The first GPU test")
p("Job 64988643 confirmed that Slurm could allocate one A100 GPU and PyTorch could perform a GPU matrix calculation. This was an environment test, not an Inkling test or a quantization. The node exposed a 64 GiB A100, driver 535.274.02, glibc 2.28, Python 3.11.6, PyTorch 2.2.0a0, Transformers 4.38.0 and CUDA 12.1.")
p("A site-managed Slurm plugin printed a permission error. The GPU calculation still passed. Later successful jobs also showed this message. Its appearance was not used as a reason to disable a security setting or alter CINECA's installation.")

page("The conversion we actually ran")
h("Why the environment changed")
p("The installed CINECA stack was sufficient for the initial GPU test but too old for the chosen model tools. Current MLX Linux wheels require glibc 2.35 or newer; the documented CUDA wheel also requires a driver newer than Leonardo's observed driver. We used a Python 3.11 Debian Bookworm container to supply newer user-space libraries and installed CPU MLX inside it. A container cannot upgrade the host GPU driver.")
p("Initial research found an Inkling 2 Small pull request, which did not establish Mini support. A Mini-specific port subsequently became available as MLX-LM pull request 2086. We pinned that implementation rather than assuming a normal released installation would work. A separate community mixed 2-bit checkpoint was found, with a reported size of 13.24 GB. It was context for comparison, not our source model or an accuracy baseline.")
table(["Component", "Our stored precision", "Reason"], [
    ("All routed expert matrices", "2-bit affine", "The largest opportunity to reduce weight storage"),
    ("Other eligible linear matrices including attention", "4-bit affine", "A less aggressive starting point for smaller components"),
    ("Token embeddings and output head", "8-bit affine", "Protect input and output representations"),
    ("Vision encoder", "Source precision", "Preserve image processing capability"),
    ("Routing and loop controllers", "Source precision", "Preserve expert selection and cross-pass control"),
    ("Norms convolutions and incompatible widths", "Source precision", "Retain the original numerical representation"),
], [2.35, 1.45, 3.2])
p("The affine group size was 64, with BF16 scale and offset metadata. Both original NVFP4 experts and MXFP8 tail experts were repacked. The converter reconstructed at most one expert or a dense-matrix slice at a time rather than creating a full BF16 model. This is weight-only round-to-nearest quantization. No activation calibration, GPTQ, AWQ, fine-tuning or training was performed.")
h("Pinned versions and source")
code("MLX 0.32.4 with the CPU backend\nMLX-LM commit\n0c4c99192230f1943dd468d297f0b8a64ae7b7c2\nRequested original source revision\n1fb4134b9a74f9681054f4ef82e4bf1f458a2337")
p("The original weights remained unchanged. The vision encoder, image processor, chat template and tokenizer assets were retained. The optional DSpark speculative-decoding model was excluded. Nothing was published or uploaded to a model hosting service.")

page("What was verified and what was not")
table(["Check", "Evidence"], [
    ("Original expert formats", "Small tests reconstructed NVFP4 with per-expert scales and MXFP8, then checked expected 2-bit packed tensors"),
    ("Numerical edge cases", "Zero and constant rows stayed finite; scalar and transposed tensors could be fingerprinted"),
    ("Loop and image behavior in a tiny model", "A synthetic model converted, reloaded, and ran text and image forward passes"),
    ("Full checkpoint structure", "24 stored layers, 48 layer applications, 48 expert stacks and 128 experts per stack"),
    ("Full saved tensor contents", "1567 matching SHA256 signatures, shapes and dtypes after strict reload"),
    ("Unmodified components", "Their tensors and copied tokenizer/image assets were checked for exact preservation"),
], [2.3, 4.7])
h("The recorded jobs")
table(["Job", "Purpose", "Outcome"], [
    ("64988643", "Initial GPU environment test", "GPU calculation passed"),
    ("64990371", "First full launch attempt", "Stopped after 7 seconds before conversion because my launcher checked a container Python symlink on the host"),
    ("64990374", "Linux container compatibility test", "Completed successfully; the original three tests passed"),
    ("64991140", "Corrected conversion and four tests", "Completed with exit code 0 in 8 minutes 33 seconds"),
], [1.0, 2.3, 3.7])
p("The launcher error was corrected by checking the virtual environment configuration on the host and executing Python inside the container. Scheduling forecasts initially suggested long waits, but the jobs started sooner; those forecasts were not guarantees. The final job requested eight CPU cores, 96 GiB host memory, no GPU and a 30-minute limit. Its conversion step reached roughly 15.9 GiB peak resident memory, according to Slurm accounting.")
h("How to interpret success")
p("Exact file reload checks establish that serialization did not change the intended tensors. They do not establish accuracy. Small synthetic tests establish that selected code paths work; they are not benchmarks of the 26B model. The per-matrix reconstruction errors in the conversion report are weight statistics, not scores for reasoning, coding or image understanding.")
p("The final output has 48 matrices stored at 2 bits, 176 at 4 bits and two at 8 bits, plus the preserved components. Weight files total 11019161398 bytes. Working memory during inference also includes caches, image activations and temporary buffers, while macOS and other applications need their own RAM.")

page("Download and local Mac measurements")
h("Local copy and integrity")
if download.get("status") == "passed":
    p(f"The checkpoint was downloaded to this Mac. A streaming verification compared all {download['verified_tensor_count']} local tensor hashes, shapes and dtypes with the post-conversion signatures from Leonardo. All matched. Verification read the safetensors data in small chunks, so it did not require loading the model into GPU memory.")
else:
    p("The checkpoint download and its streamed integrity check are in progress. This section will be updated with the final result.")
p("The deployment machine is an Apple M5 with 16 GiB unified memory and macOS 26.6.2. MLX reports a recommended GPU working set of about 12.71 GB. During preparation the machine showed warning-level memory pressure. Immediately before the first probe it had returned to normal pressure level 1, with 13874.81 M of swap already used according to sysctl. After the four probes it showed warning level 2 and 19975.31 M of swap used, an increase of roughly 6 GiB. This is system-wide swap, including background applications, not a measurement of model weights spilling to disk alone.")
h("Test method")
p("The smoke tests used greedy sampling, the supplied chat template with reasoning effort set to none, a 64-token prefill step and a 192-token output ceiling. They covered arithmetic, extracting an order number, a small Python function and a 240 by 160 pixel image with two colored shapes. MLX_ENABLE_TF32 was set to 0 for the documented full float32 path on M5. The model still generated reasoning before its final answer, so generation speeds include those reasoning tokens. The table shows only the final answer. These tests do not measure retained accuracy against the original model.")
if probe:
    p("Recorded test state: " + str(probe.get("state", "unknown")) + ".")
    if probe.get("error"):
        p("Recorded issue: " + probe["error"])
    rows = []
    for case in probe.get("cases", []):
        answer = case.get("output", "").rsplit("<|content_text|>", 1)[-1].split("<|end_message|>", 1)[0]
        rows.append((case["case"], f"{case.get('generation_tokens_per_second', 0):.2f}",
                     f"{case.get('peak_mlx_bytes', 0)/2**30:.2f}",
                     answer.replace("\n", " ")[:180]))
    if rows:
        table(["Case", "Output tokens per second", "Peak MLX GiB", "Observed output excerpt"], rows, [0.9, 1.0, 0.9, 4.2])
    if probe.get("watchdog"):
        p("Resource guard: " + str(probe["watchdog"]))
    if probe.get("cases"):
        p(f"Loading took {probe.get('load_seconds', 0):.1f} seconds. The first arithmetic prompt then took 34.4 seconds to its first generated token; the two subsequent text prompts took about 1.0 and 0.6 seconds. The first image prompt took 23.2 seconds to its first token. Cold loading, compilation and memory pressure can affect latency. Peak MLX allocation reached about 10.53 GiB. This figure excludes other applications and some host memory; process RSS alone does not capture Metal memory use.")
        p("The separate ready-to-use launcher also returned the correct image description. That check took 107 seconds overall at about 28 generated tokens per second while report processing was also active. It was a functionality check under busy conditions, not a controlled timing benchmark. End-to-end waiting time can be much longer than the decoding rate suggests.")
else:
    p("Local inference measurements are pending. No speed, quality or comfortable 16 GB fit claim is made at this point.")
h("What the Mac result means")
p("A sensible short answer establishes basic functionality only. A model can load and still be too slow, use substantial swap or fail on longer contexts and larger images. The JSON results retain complete generated text, timing, prompt lengths and memory observations so the conclusion can be revisited. Background applications and cold loading can affect these measurements.")

if context.get("cases"):
    h("Longer prompts and first use overhead")
    p("A synthetic retrieval probe put LANTERN-42 near the beginning of a note, repeated irrelevant background text, and asked for the identifier. It returned the identifier correctly at all three measured lengths. This is a simple retention check, not a long-context benchmark. Each test used a fresh attention cache.")
    table(["Actual prompt tokens", "First token seconds", "Output tokens per second", "Peak MLX GiB"], [
        (c.get("prompt_tokens"), f"{c.get('first_token_seconds',0):.1f}",
         f"{c.get('generation_tokens_per_second',0):.2f}", f"{c['peak_mlx_bytes']/2**30:.2f}")
        for c in context["cases"]], [1.6, 1.6, 1.9, 1.9])
    p("The larger prompt tests were slower than the short probes. In the first 4088-token run, generation averaged only about 3.5 tokens per second. A separate tuning pass checks whether larger prefill batches and repeated execution improve that result. Do not extrapolate the short-prompt 25 to 29 token rate to all document workloads.")
if tuning.get("cases"):
    h("Prefill tuning follow up")
    p("The follow-up used a 256-token prefill step instead of 64 and repeated the long prompt in the same process. The table preserves the trial order. These sequential measurements were made on the same busy Mac, not under controlled idle conditions; batch size, compilation and changing memory pressure can all contribute to the differences.")
    table(["Trial", "Prompt tokens", "First token seconds", "Output tokens per second", "Peak MLX GiB"], [
        (i+1, c.get("prompt_tokens"), f"{c.get('first_token_seconds',0):.1f}",
         f"{c.get('generation_tokens_per_second',0):.2f}", f"{c['peak_mlx_bytes']/2**30:.2f}")
        for i,c in enumerate(tuning["cases"])], [.6, 1.3, 1.6, 1.8, 1.7])
    if tuning.get("error") or tuning.get("watchdog"):
        p("Tuning issue: " + str(tuning.get("error") or tuning.get("watchdog")))

page("Using the results and deciding the next experiment")
h("Where the work is saved")
p("The local project root is:")
code(str(ROOT))
table(["File or folder inside outputs", "Purpose"], [
    ("Inkling-2-Mini-expert-2bit-g64", "The downloaded local model checkpoint"),
    ("CONVERSION_REPORT.json", "Recipe, per-matrix diagnostics and full tensor signatures"),
    ("DOWNLOAD_VERIFICATION.json", "Local integrity verification"),
    ("MAC_INFERENCE_RESULTS.json", "Complete local smoke-test measurements and generated outputs"),
    ("inkling-conversion-kit.zip", "Reproducible conversion code, tests and Slurm job script"),
    ("mac_probe.py", "The bounded local inference test program"),
    ("run_inkling.command and ask_inkling.py", "A local text and image launcher with no network server"),
    ("HOW_TO_RUN.txt", "Instructions for starting the model and rebuilding its environment"),
], [3.0, 4.0])
p("The Leonardo checkpoint remains in:")
code("/PROJECT/converted/\nInkling-2-Mini-expert-2bit-g64-64991140")
h("The next decision")
p("Double-click run_inkling.command in outputs to start. Enter a question and optional image path; /quit releases the model. Each question starts fresh. The launcher caps total context at 4096 tokens and images at 512 pixels on the longer side. It runs offline. HOW_TO_RUN.txt explains command-line use.")
p("The basic probes work, but comfortable 16 GB use remains conditional. Retest with memory-heavy applications closed, then evaluate real documents, OCR and charts. Measure prompt processing and generation separately, counting reasoning tokens. Only this M5 Mac was tested.")
p("For better quality, compare calibration and selective higher precision on held-out tasks against the original model. Group size 32 may improve fidelity at greater storage cost; 128 may save memory at greater error. If swapping dominates, compare a smaller model or a more compact recipe.")
p("Leonardo can help optimize compression but cannot increase the Mac's memory. The allocation is listed to end on 24 November 2026. Preserve local weights, code and reports independently of that deadline.")
h("Sources and primary evidence")
link("Thinking Machines original checkpoint and model card", "https://huggingface.co/thinkingmachines/Inkling-2-Mini")
link("Mini MLX implementation and stated model support", "https://github.com/ml-explore/mlx-lm/pull/2086")
link("MLX installation requirements", "https://ml-explore.github.io/mlx/build/html/install.html")
link("MLX numerical precision and the M5 precision switch", "https://ml-explore.github.io/mlx/build/html/usage/precision.html")
link("CINECA authentication and host identity troubleshooting", "https://docs.hpc.cineca.it/general/access.html")
link("CINECA containers and job environment", "https://docs.hpc.cineca.it/services/singularity.html")

doc.core_properties.title = "Inkling 2 Mini quantization experiment"
doc.core_properties.subject = "Experiment history and local deployment results"
doc.core_properties.author = "Codex"
# Remove the default template's decorative title border, including inheritance.
for style in doc.styles:
    for border in list(style.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
for paragraph in doc.paragraphs:
    for border in list(paragraph._p.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
doc.save(OUT / "Inkling_experiment_report.docx")
print(OUT / "Inkling_experiment_report.docx")
