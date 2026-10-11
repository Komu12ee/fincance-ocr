# FG: recursive GPU OCR to searchable PDFs

Run `gpu_ocr_pdf.py` on the VM holding the PDFs. OCR inference runs on the other
GPU VM through an SSH tunnel; no local CUDA installation is required.
This FG command defaults to `finance_cg_data` and writes only to
`finance_cg_data_ocr`. It does not load `pipeline_config.py`, the document
registry, embedding/indexing pipelines, or any other corpus configuration.
Use Python 3.10 or newer.
The GPU server must offer an OpenAI-compatible **vision** endpoint at
`/v1/chat/completions`, accepting base64 `image_url` input. A `/v1/models` listing
alone does not prove image input is supported.

This command reuses `01_preprocessing/stage1_image_prep/pdf_to_image.py` for
page rendering and `01_preprocessing/stage2_ocr/models.py` for the existing
`DocumentOCRResult`, `PageOCRResult`, elements and bounding boxes. It writes
the same structured JSON/Markdown schema alongside each PDF. Package entry
points load their full pipelines lazily so this remote client can use those
components without installing Docling, EasyOCR or OpenCV on the processing VM.
Existing `run_stage1.py` and `run_stage2.py` entry points remain available.

## Run on the processing VM (Linux)

From the project directory, install the small Python client dependencies:

```bash
python3 -m venv .venv-ocr
source .venv-ocr/bin/activate
python -m pip install -r core/requirements_gpu_ocr.txt
```

In a separate terminal **on the processing VM**, start the tunnel to the GPU VM
and leave it running:

```bash
ssh -N -L 8000:127.0.0.1:8001 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 ubuntu@10.71.12.1
```

The GPU service must already be listening on `127.0.0.1:8001` on the GPU VM.
The processing VM must be able to reach that VM's SSH address.

Back in the activated Python terminal:

The command automatically loads `core/.env`, which contains the FG GPU URL and
model. Existing shell environment variables override this file. The exports
below are optional for the Python command; they also configure the curl check.

```bash
export GPU_OCR_BASE_URL=http://127.0.0.1:8000/v1
export GPU_OCR_MODEL=ocr-primary
curl --fail -sS "$GPU_OCR_BASE_URL/models"

# Preview recursive discovery, without calling the GPU:
python core/gpu_ocr_pdf.py finance_cg_data --dry-run

# Check one complete PDF before launching the batch:
python core/gpu_ocr_pdf.py finance_cg_data --limit 1 --workers 1

# Process every PDF recursively (including .PDF and mixed-case extensions):
python core/gpu_ocr_pdf.py finance_cg_data --workers 2
```

## Count and track FG OCR

The script recursively counts **all** source PDFs at startup. `--limit 1`
selects one PDF to process; the total still includes the full corpus. A status
panel prints at startup and after each PDF, and progress is saved before waiting
for GPU inference and after verifying each page.

To count PDFs and check status at any time, including before your first OCR run:

```bash
python core/gpu_ocr_pdf.py finance_cg_data --status
```

During OCR, open another terminal in the same project and virtual environment:

```bash
# Refresh the status panel every 3 seconds. Ctrl+C stops only this watcher.
python core/gpu_ocr_pdf.py finance_cg_data --watch

# Optional: choose a refresh interval in seconds.
python core/gpu_ocr_pdf.py finance_cg_data --watch 5
```

For custom output roots, pass the same `-o` to both the OCR and status commands.
The status commands make no GPU requests and do not write or modify results.
They recount the source PDFs and validate completion records using the last
run's OCR settings. A terminal watch refreshes the screen; redirected watch
output produces successive plain-text snapshots.

Example display (illustrative numbers):

```text
-- FG OCR STATUS --
Corpus   PDFs present   OCR done   OCR failed    Pending
fg               5058        542            2       4514
Progress: [###---------------------------] 10.7%
Last run: RUNNING
This run: selected 5058 | checked 12 | newly done 10 | already done 0 | failed 2
Pages verified this run: 31 (cached: 4)
Current PDF [13/5058]: 02_Rules_And_Act/1982/Amend_2002.pdf
Current page: 2/7 | waiting for GPU OCR
GPU API: http://127.0.0.1:8000/v1 | last check: ready
Elapsed: 00:18:42 | Free disk: 1823.0 GiB
```

- **PDFs present:** every PDF currently found recursively under the FG input.
- **OCR done:** completed PDFs whose source, configuration, PDF writer version,
  output PDF size/mtime and required sidecars still match the completion record.
- **OCR failed:** PDFs with a current tracked failure. Successful retries clear
  that failure count. The append-only `errors.jsonl` remains a history.
- **Pending:** PDFs still to finish, including the one currently in progress.
  `done + failed + pending` equals the total. Changed sources or missing outputs
  return to pending. Previous results that need a writer upgrade also do so.
- **This run:** counts for the current selection, separate from corpus totals.
  Completing a limited run does not mean the whole corpus has finished.
- **GPU API:** the result of the last endpoint check, with its UTC timestamp;
  this does not measure GPU memory or utilization on the other VM.

`finance_cg_data_ocr/ocr_status.json` is a machine-readable snapshot. Per-document
`progress.json` records live in `.ocr-cache/<document-hash>/`. On Linux, the
watcher identifies an OCR process that exited without updating its checkpoint.
Rerun OCR to resume; watching never starts or stops an OCR job.

For long runs, use a persistent terminal such as tmux. Logs can be redirected:

```bash
python -u core/gpu_ocr_pdf.py finance_cg_data --workers 2 > gpu_ocr.log 2>&1
```

## Output

```text
finance_cg_data/category/subfolder/report.pdf
    -> finance_cg_data_ocr/category/subfolder/report.pdf
```

- Same PDF filenames and relative directories; originals remain untouched.
- Same page count, page dimensions and visible content rendered at 200 DPI by
  default. Pages are rebuilt from images, removing broken/legacy original text
  layers. This rasterizes vector content and annotations; interactive forms,
  original hyperlinks and digital signatures are not carried into the output.
- Invisible Unicode OCR text: select/copy text or use your PDF viewer's Find
  command (Ctrl+F). Hindi uses PyMuPDF's bundled Devanagari font explicitly,
  preserving sentence punctuation (`।`, `॥`); English uses Helvetica.
- Original outline titles plus a bookmark for every page; page labels retained.
- `finance_cg_data_ocr/index.html`: a browsable file index with path filtering.
  It filters filenames/paths; search document contents inside a PDF viewer.
- `.ocr-cache/`: successful page OCR and completion records for resuming.
- `report.pdf.ocr.json` and `report.pdf.ocr.md` beside each output PDF: reusable
  Stage 2 structured results; JSON bounding boxes use rendered-image pixels.
- `errors.jsonl`: appended failures; exit status 1 if any PDF failed, 2 if the
  GPU readiness check failed. A failed PDF is not published as a partial result.
  Unicode text-layer failures also retain expected/extracted text in
  `.ocr-cache/<document-hash>/text-layer-failure-000001.json` for diagnosis.

Rerun the same command to resume. Unchanged completed PDFs are skipped; partial
PDFs reuse successful page responses. Source size/mtime and model, prompt,
rendering and token settings identify the cache. Use `--force` if inputs were
changed while retaining their original size and timestamp, or to redo OCR.
Text-layer writer upgrades rebuild PDFs using cached OCR without repeating GPU
inference. After updating the code, rerun normally; `--force` is not necessary.
Use a single process per output directory to avoid conflicting cache writes.

To choose another output root:

```bash
python core/gpu_ocr_pdf.py finance_cg_data -o /data/searchable_pdfs
```

## Model compatibility and tuning

The default prompt asks the GPU model for line text and bounding boxes normalized
to 0..1000. Bounding boxes let PDF search highlight the correct page regions.
If the model supplies only plain text, the script uses approximate positions:
the PDF is searchable, but selection/highlights may not line up with the scan.
OCR accuracy and coordinate accuracy still need checking on your model.

If your model requires a specific OCR prompt, place it in a UTF-8 file and run:

```bash
python core/gpu_ocr_pdf.py finance_cg_data --prompt-file /data/ocr_prompt.txt
```

Prompt overrides can return the documented JSON or plain transcription. For a
blank page, return `{"blank": true, "lines": []}`. Empty responses, invalid
coordinates and incomplete completions fail instead of silently losing text.
Models emitting proprietary grounding tags need a prompt requesting this JSON
or plain text format.

- Start with `--workers 1`; increase only if the GPU service supports parallel
  inference. Rendering/PDF writing stay serial; HTTP requests run concurrently.
- Use `--dpi 300 --max-pixels 12000000` for small print, within server limits.
- Increase `--max-tokens` for dense pages only when the server context has room
  for both the image/prompt tokens and the requested output. Requesting 16384
  output tokens from a server with a 16384-token total context leaves no room
  for input and can cause HTTP 400.
- Use `--timeout 300` for slower inference.
- Set `GPU_OCR_API_KEY` if the GPU service requires bearer authentication.

### HTTP 400: inspect the server's reason

The client prints the server's error message, type, parameter and code, and saves
the same reason in `errors.jsonl` and the status tracker. HTTP 400 means the
server rejected the request; an accessible `/v1/models` endpoint alone cannot
establish why. The script does not retry identical HTTP 400 requests or switch
OCR engines.

For a first test with a smaller image and output budget:

```bash
python core/gpu_ocr_pdf.py finance_cg_data --limit 1 --workers 1 --dpi 150 --max-pixels 1500000 --max-tokens 4096
```

If the error names the context length, reduce the input image/output budget or
increase the GPU server's configured context if resources permit. If it names
disabled images, unsupported image input or a chat-template problem, fix that
GPU server configuration; smaller images cannot enable vision support. Keep
the complete server error when reporting a failure. Reduced image resolution
can impair small-print OCR, and a `finish_reason=length` response still fails
rather than publishing truncated text. Changed rendering/token settings use
a different cache configuration; reuse the same settings when resuming.

The code uses PyMuPDF's [invisible text rendering mode](https://pymupdf.readthedocs.io/en/latest/textwriter.html)
for the searchable layer.

## Share FG VM diagnostics through GitHub

From the processing VM, with the OCR environment activated and GPU tunnel open:

```bash
git pull --ff-only origin main
python core/fg_vm_diagnostics.py finance_cg_data --probe --push
```

This collects package versions, the checked-out commit, FG PDF counts, recent
OCR errors, status and model metadata. `--probe` sends the first PDF's first
page to the configured GPU once with the last run's rendering/token settings
(or defaults before the first run). It records the server's full rejection
reason without uploading the page or resulting OCR text to GitHub. It uses
the default OCR prompt; a job's custom `--prompt-file` is not reproduced.
Neither collection nor the probe changes existing OCR outputs or page caches.
This probe can take up to 180 seconds of read time; keep the terminal open.

`--push` commits only the generated report to a new `fg-ocr-diagnostics-*`
branch using an isolated temporary Git checkout. Your current VM branch,
staged changes and `main` are untouched. GitHub authentication uses the same
`origin` URL and SSH configuration as your existing repository; write access
is required. No force push is used. If upload fails, the local report remains.
The script prints the uploaded branch name and link. Share that link in chat
so the report can be fetched and inspected.

Reports contain document filenames and error messages. They exclude source
PDFs, OCR transcripts, full text-layer diagnostics, arbitrary VM logs, `.env`
contents and credentials. API keys and common token patterns in messages are
redacted. Run without `--push` to inspect a local report first, and omit
`--probe` to collect existing errors without invoking GPU inference. Local
reports under `diagnostics/fg-vm-*.json` are ignored by Git; only this script's
explicit upload stages the single report. For a custom OCR output directory,
pass the matching `-o /path/to/output`.
