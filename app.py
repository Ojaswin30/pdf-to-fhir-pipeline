"""
app.py - Flask web application for the GraphRAG pipeline.
Run with: python app.py
"""

import os
import json
import glob
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

from config import PDF_FOLDER, FLASK_SECRET_KEY, FLASK_PORT, FLASK_DEBUG
from main import GraphRAG, get_pdf_files_by_patient

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

UPLOAD_FOLDER = PDF_FOLDER
ALLOWED_EXTENSIONS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("fhir", exist_ok=True)
os.makedirs("vector_store", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload

# ---------- Global state ----------
processing_queues: dict = {}  # job_id -> queue.Queue

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Routes ----------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/pdfs", methods=["GET"])
def list_pdfs():
    """List all PDFs grouped by patient."""
    patient_groups = get_pdf_files_by_patient(PDF_FOLDER)
    result = {}
    for patient_id, paths in patient_groups.items():
        result[patient_id] = [os.path.basename(p) for p in paths]
    return jsonify({"patients": result})

@app.route("/api/fhir_outputs", methods=["GET"])
def list_fhir_outputs():
    """List all generated FHIR JSON files."""
    files = glob.glob("fhir/combined_*.json")
    outputs = []
    for f in files:
        patient_id = os.path.basename(f).replace("combined_", "").replace(".json", "")
        outputs.append({"patient_id": patient_id, "file": os.path.basename(f)})
    return jsonify({"outputs": outputs})

@app.route("/api/fhir/<patient_id>", methods=["GET"])
def get_fhir(patient_id):
    """Get FHIR JSON for a specific patient."""
    path = f"fhir/combined_{patient_id}.json"
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    with open(path) as f:
        return jsonify(json.load(f))

@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    """Upload one or more PDF files."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist("files")
    uploaded = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(dest)
            uploaded.append(filename)
        else:
            errors.append(f"Rejected: {file.filename}")
    
    return jsonify({"uploaded": uploaded, "errors": errors})

@app.route("/api/process", methods=["POST"])
def process_pdfs():
    """
    Start processing PDFs for a given patient (or all patients).
    Returns a job_id to stream logs from /api/logs/<job_id>.
    """
    data = request.json or {}
    patient_id = data.get("patient_id")  # None = process all

    import uuid
    job_id = str(uuid.uuid4())
    log_queue = queue.Queue()
    processing_queues[job_id] = log_queue

    def run_processing():
        def log(msg):
            print(msg)
            log_queue.put(msg)

        try:
            patient_groups = get_pdf_files_by_patient(PDF_FOLDER)

            if not patient_groups:
                log(f"[ERROR] No PDF files found in {PDF_FOLDER}")
                return

            targets = {patient_id: patient_groups[patient_id]} if patient_id and patient_id in patient_groups else patient_groups

            log(f"[INFO] Found {len(targets)} patient(s) to process")

            for pid, pdf_paths in targets.items():
                log(f"\n{'='*60}")
                log(f"PROCESSING PATIENT: {pid}")
                log(f"{'='*60}")
                rag = GraphRAG(pid)
                rag.ingest_pdfs(pdf_paths, log_fn=log)
                rag.graph.close()
                log(f"[INFO] Completed processing for patient {pid}")

            log("[DONE] All patients processed successfully.")
        except Exception as e:
            log(f"[FATAL] Processing failed: {e}")
        finally:
            log_queue.put(None)  # Sentinel: stream done

    thread = threading.Thread(target=run_processing, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})

@app.route("/api/logs/<job_id>")
def stream_logs(job_id):
    """Server-Sent Events stream for live logs."""
    if job_id not in processing_queues:
        return jsonify({"error": "Job not found"}), 404

    log_queue = processing_queues[job_id]

    def generate():
        while True:
            msg = log_queue.get()
            if msg is None:
                yield "data: [DONE]\n\n"
                break
            # Escape for SSE
            for line in msg.split("\n"):
                yield f"data: {line}\n"
            yield "\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

@app.route("/api/query", methods=["POST"])
def query():
    """Run a natural language query against a patient's data."""
    data = request.json or {}
    patient_id = data.get("patient_id", "").strip()
    question = data.get("question", "").strip()

    if not patient_id or not question:
        return jsonify({"error": "patient_id and question are required"}), 400

    try:
        rag = GraphRAG(patient_id)
        answer = rag.run(question)
        rag.graph.close()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Run ----------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True
    )