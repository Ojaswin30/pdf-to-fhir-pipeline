import os
import time
import pickle
import fitz  # PyMuPDF for layout-aware parsing
import ocrmypdf
import tempfile
import glob
import re
from typing import List, Any, Dict
import importlib
import sys

# Robust dynamic imports to handle differing langchain installs
def import_attr(attr_name: str, candidates: list):
    """Try to import `attr_name` from a list of module paths in order."""
    for mod_path in candidates:
        try:
            module = importlib.import_module(mod_path)
            if hasattr(module, attr_name):
                return getattr(module, attr_name)
        except Exception:
            continue
    return None

# Try common locations for these symbols (fallbacks ordered by preference)
LLMChain = import_attr("LLMChain", [
    "langchain.chains",
    "langchain.chains.llm",
    "langchain_classic.chains.llm",
])

PromptTemplate = import_attr("PromptTemplate", [
    "langchain.prompts",
    "langchain.prompts.prompt",
    "langchain_core.prompts",
])

# langchain_together/Together
try:
    Together = importlib.import_module("langchain_together").Together
except Exception:
    Together = None

# FAISS vectorstore from community package
FAISS = import_attr("FAISS", [
    "langchain_community.vectorstores",
    "langchain_community.vectorstores.faiss",
])

# HuggingFace embeddings
HuggingFaceEmbeddings = import_attr("HuggingFaceEmbeddings", [
    "langchain_huggingface",
    "langchain_huggingface.huggingface",
])

# Document and text splitter
Document = import_attr("Document", [
    "langchain.docstore.document",
    "langchain.schema.document",
])

RecursiveCharacterTextSplitter = import_attr("RecursiveCharacterTextSplitter", [
    "langchain.text_splitter",
    "langchain.text_splitter.recursive",
    "langchain.text_splitter.text_splitter",
])

# Additional fallbacks for the text splitter package
if RecursiveCharacterTextSplitter is None:
    try:
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter as _RCTS
        RecursiveCharacterTextSplitter = _RCTS
    except Exception:
        try:
            from langchain_classic.text_splitter import RecursiveCharacterTextSplitter as _RCTS2
            RecursiveCharacterTextSplitter = _RCTS2
        except Exception:
            RecursiveCharacterTextSplitter = None
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
import json
from collections import defaultdict

# ---------- Load config from config.py (secrets from .env) ----------
from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    TOGETHER_API_KEY, PDF_FOLDER
)

# ---------- Advanced Targeted OCR Processing ----------
def create_image_only_pdf_for_ocr(input_pdf_path: str) -> str:
    """
    Create a PDF containing only images (no text) for targeted OCR processing.
    Then merge the OCR results back with the original text.
    """
    temp_images_pdf = tempfile.mktemp(suffix="_images_only.pdf")
    
    try:
        print("[INFO] Creating image-only PDF for targeted OCR...")
        
        doc = fitz.open(input_pdf_path)
        new_doc = fitz.open()
        
        pages_with_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            new_page = new_doc.new_page(width=rect.width, height=rect.height)
            
            image_list = page.get_images(full=True)
            
            if image_list:
                print(f"[INFO] Page {page_num + 1}: Found {len(image_list)} images")
                pages_with_images.append(page_num)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                            img_rects = page.get_image_rects(xref)
                            
                            for img_rect in img_rects:
                                new_page.insert_image(img_rect, stream=img_data)
                        
                        pix = None
                        
                    except Exception as e:
                        print(f"[WARN] Failed to process image {img_index} on page {page_num}: {e}")
        
        new_doc.save(temp_images_pdf)
        new_doc.close()
        doc.close()
        
        print(f"[INFO] Created image-only PDF with {len(pages_with_images)} pages containing images")
        return temp_images_pdf, pages_with_images
        
    except Exception as e:
        print(f"[ERROR] Failed to create image-only PDF: {e}")
        if os.path.exists(temp_images_pdf):
            os.remove(temp_images_pdf)
        return None, []

def apply_targeted_ocr_with_ocrmypdf(input_pdf_path: str) -> str:
    ocr_output_path = input_pdf_path.replace(".pdf", "_targeted_ocr.pdf")
    
    if os.path.exists(ocr_output_path):
        print("[INFO] Targeted OCR already applied, using cached version.")
        return ocr_output_path

    print("[INFO] Applying targeted OCR using advanced image extraction...")
    
    temp_images_pdf, pages_with_images = create_image_only_pdf_for_ocr(input_pdf_path)
    
    if not temp_images_pdf or not pages_with_images:
        print("[INFO] No images found for OCR, using original PDF")
        return input_pdf_path
    
    temp_ocr_pdf = tempfile.mktemp(suffix="_temp_ocr.pdf")
    
    try:
        print(f"[INFO] OCR processing {len(pages_with_images)} pages with images...")
        
        ocrmypdf.ocr(
            temp_images_pdf,
            temp_ocr_pdf,
            force_ocr=True,
            language=['eng'],
            use_threads=True,
            optimize=1,
            tesseract_timeout=300,
            jpeg_quality=95,
            png_quality=95,
        )
        
        final_output = merge_original_with_ocr_results(
            input_pdf_path, 
            temp_ocr_pdf, 
            ocr_output_path,
            pages_with_images
        )
        
        print(f"[INFO] Targeted OCR complete: {final_output}")
        return final_output
        
    except Exception as e:
        print(f"[ERROR] Targeted OCR failed: {e}")
        return input_pdf_path
        
    finally:
        for temp_file in [temp_images_pdf, temp_ocr_pdf]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def merge_original_with_ocr_results(original_pdf: str, ocr_pdf: str, output_pdf: str, pages_with_images: list) -> str:
    try:
        print("[INFO] Merging original PDF with OCR results...")
        
        original_doc = fitz.open(original_pdf)
        ocr_doc = fitz.open(ocr_pdf)
        merged_doc = fitz.open()
        
        for page_num in range(len(original_doc)):
            original_page = original_doc[page_num]
            rect = original_page.rect
            new_page = merged_doc.new_page(width=rect.width, height=rect.height)
            
            new_page.show_pdf_page(rect, original_doc, page_num)
            
            if page_num in pages_with_images and page_num < len(ocr_doc):
                ocr_page = ocr_doc[page_num]
                
                try:
                    ocr_text_dict = ocr_page.get_text("dict")
                    
                    for block in ocr_text_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line.get("spans", []):
                                    text = span.get("text", "").strip()
                                    if text:
                                        bbox = span.get("bbox")
                                        if bbox:
                                            new_page.insert_text(
                                                (bbox[0], bbox[1]), 
                                                text, 
                                                fontsize=span.get("size", 12),
                                                render_mode=3
                                            )
                except Exception as e:
                    print(f"[WARN] Failed to add OCR text for page {page_num + 1}: {e}")
        
        merged_doc.save(output_pdf)
        merged_doc.close()
        original_doc.close()
        ocr_doc.close()
        
        return output_pdf
        
    except Exception as e:
        print(f"[ERROR] Failed to merge PDFs: {e}")
        return original_pdf

def get_comprehensive_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        
        for page_num, page in enumerate(doc):
            print(f"[INFO] Extracting text from page {page_num + 1}")
            
            page_text = page.get_text()
            text_dict = page.get_text("dict")
            additional_text = []
            
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    block_text = []
                    for line in block["lines"]:
                        line_text = []
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                line_text.append(text)
                        if line_text:
                            block_text.append(" ".join(line_text))
                    if block_text:
                        additional_text.append("\n".join(block_text))
            
            if additional_text:
                combined_additional = "\n".join(additional_text)
                if len(combined_additional) > len(page_text) * 1.1:
                    page_text = combined_additional
            
            if page_text.strip():
                print(f"[DEBUG] Page {page_num + 1}: {len(page_text)} characters extracted")
                full_text.append(page_text)
            else:
                print(f"[WARN] No text found on page {page_num + 1}")
        
        doc.close()
        
        combined_text = "\n\n".join(full_text)
        print(f"[INFO] Total extracted text: {len(combined_text)} characters")
        
        return combined_text
        
    except Exception as e:
        print(f"[ERROR] Failed to extract text: {e}")
        return ""

# ---------- Retry Decorator ----------
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
def safe_llm_call(fn, *args, **kwargs):
    time.sleep(1)
    return fn(*args, **kwargs)

# ---------- Pydantic Schema ----------
class QueryInput(BaseModel):
    question: str = Field(..., description="The user's natural language question")

# ---------- Initialize LLM ----------
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=512,
    together_api_key=TOGETHER_API_KEY
)

# Determine whether to use remote LLMs or fall back to local, deterministic logic
USE_LLM = bool(TOGETHER_API_KEY and Together)

def local_summarize(texts: list, question: str = "") -> str:
    """Simple local summarizer fallback: join first few chunks and return trimmed text."""
    if not texts:
        return "No text available for summary."
    joined = "\n\n".join(texts[:3])
    snippet = joined[:1500]
    return f"[Local summary]\n{snippet}\n\n(Truncated - enable an LLM for richer summaries)"

def local_extract_triples_text(text: str) -> str:
    """Extract basic triples from structured fields like Diagnosis / Recommendations."""
    triples = []
    info = extract_patient_info(text)
    patient = info.get("name", "Patient") or "Patient"
    if info.get("diagnosis"):
        triples.append(f"({patient}) -[HAS_DIAGNOSIS]-> ({info['diagnosis']})")
    if info.get("history"):
        triples.append(f"({patient}) -[HAS_HISTORY]-> ({info['history']})")
    if info.get("recommendations"):
        triples.append(f"({patient}) -[HAS_RECOMMENDATION]-> ({info['recommendations']})")
    if info.get("physical_exam"):
        triples.append(f"({patient}) -[HAS_PHYSICAL_EXAM]-> ({info['physical_exam']})")

    # As a fallback, try to find simple "A: B" lines
    for line in text.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key and val and len(val) < 200:
                triples.append(f"({patient}) -[{key.upper().replace(' ', '_')}]-> ({val})")

    if not triples:
        return ""
    return "\n".join(triples)

# ---------- Local LLM support (llama_cpp or gpt4all) ----------
USE_LOCAL_LLM = False
local_llm_model = None

def try_load_local_llm():
    global USE_LOCAL_LLM, local_llm_model
    # Try llama_cpp first
    try:
        from llama_cpp import Llama
        model_path = os.getenv("LLAMA_CPP_MODEL_PATH", "./models/llama.bin")
        if os.path.exists(model_path):
            local_llm_model = Llama(model_path=model_path, n_ctx=2048)
            USE_LOCAL_LLM = True
            print(f"[INFO] Loaded local Llama model from {model_path}")
            return
    except Exception:
        pass

    # Try gpt4all
    try:
        from gpt4all import GPT4All
        model_name = os.getenv("GPT4ALL_MODEL", "gpt4all-lora-quantized.bin")
        local_llm_model = GPT4All(model_name)
        USE_LOCAL_LLM = True
        print(f"[INFO] Loaded local GPT4All model: {model_name}")
        return
    except Exception:
        pass

try_load_local_llm()

def local_llm_call(prompt: str, max_tokens: int = 256, temperature: float = 0.3) -> dict:
    """Generate text using a locally-available LLM (returns dict with 'text')."""
    if not USE_LOCAL_LLM or local_llm_model is None:
        return {"text": ""}

    try:
        if local_llm_model.__class__.__name__.lower().startswith("llama"):
            # llama_cpp Llama
            out = local_llm_model(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            text = out["choices"][0]["text"] if isinstance(out, dict) and "choices" in out else str(out)
            return {"text": text}
        else:
            # gpt4all
            text = local_llm_model.generate(prompt)
            return {"text": text}
    except Exception as e:
        print(f"[WARN] Local LLM generation failed: {e}")
        return {"text": ""}

# ---------- Neo4j ----------
class GraphInterface:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_cypher(self, query: str) -> List[Any]:
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

# ---------- FAISS Vector Store ----------
class VectorStore:
    def __init__(self, patient_id: str):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        os.makedirs("vector_store", exist_ok=True)
        self.index_path = f"vector_store/faiss_index_{patient_id}.pkl"
        self.vectorstore = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                return pickle.load(f)
        else:
            return None

    def add_documents(self, docs: List[Document]):
        if not docs:
            print("[WARN] No documents to add to vector store")
            return
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        self._save()

    def _save(self):
        if self.vectorstore:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.vectorstore, f)

    def search(self, query: str, k: int = 5) -> List[Document]:
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

# ---------- LangChain Prompts ----------
cypher_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an assistant that translates natural language questions into Cypher queries
for a Neo4j graph database. Generate a Cypher query that answers the following question:

Question: {question}

Cypher Query:
"""
)

answer_prompt = PromptTemplate(
    input_variables=["question", "graph_results", "vector_results"],
    template="""
You are an expert assistant combining structured (graph) and unstructured (text) data
to answer user questions.

Question: {question}

Graph Data:
{graph_results}

Text Snippets:
{vector_results}

Answer the question using the above information:
"""
)

extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Extract triples (subject, relation, object) from the following text:

Text:
{text}

Return each triple in the format:
(subject) -[RELATION]-> (object)
"""
)

cypher_chain = LLMChain(llm=llm, prompt=cypher_prompt)
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
extraction_chain = LLMChain(llm=llm, prompt=extraction_prompt)

# ---------- PDF Loading ----------
def load_pdf_chunks_enhanced(path: str) -> List[Document]:
    print("[INFO] Loading PDF with enhanced text extraction...")
    
    full_text = get_comprehensive_text_from_pdf(path)
    
    if not full_text.strip():
        print("[ERROR] No text could be extracted from PDF")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.create_documents([full_text])
    print(f"[INFO] Split into {len(chunks)} chunks.")
    
    if chunks:
        print(f"[DEBUG] First chunk sample: {chunks[0].page_content[:200]}...")
    
    return chunks

# ---------- KG Construction ----------
def extract_triples_and_populate_kg(docs: List[Document], graph: GraphInterface):
    if not docs:
        print("[WARN] No documents provided for knowledge graph construction")
        return
        
    MAX_CHARS = 2000
    for i, doc in enumerate(docs):
        try:
            text = doc.page_content.strip()
            if not text:
                print(f"[WARN] Skipping empty chunk {i+1}")
                continue
                
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]

            print(f"[INFO] Processing chunk {i+1}/{len(docs)}")
            try:
                if 'extraction_chain' in globals() and extraction_chain is not None and USE_LLM:
                    response = safe_llm_call(extraction_chain.invoke, {"text": text})
                    response_text = response.get("text", "") if isinstance(response, dict) else str(response)
                    print(f"[INFO] Extracted Triples (LLM):\n{response_text}")
                    triples = parse_triples({"text": response_text})
                elif USE_LOCAL_LLM:
                    # Use local LLM to run the extraction prompt
                    prompt_text = extraction_prompt.format(text=text)
                    response = local_llm_call(prompt_text, max_tokens=512)
                    response_text = response.get("text", "")
                    print(f"[INFO] Extracted Triples (local LLM):\n{response_text}")
                    triples = parse_triples({"text": response_text})
                else:
                    # Local extraction fallback (no LLM / API key)
                    response_text = local_extract_triples_text(text)
                    print(f"[INFO] Extracted Triples (local):\n{response_text}")
                    triples = []
                    if response_text:
                        for line in response_text.splitlines():
                            if ")-[" in line and "]->(" in line:
                                try:
                                    subj, rest = line.split(")-[")
                                    rel, obj = rest.split("]->(")
                                    subj = subj.strip("() ")
                                    rel = rel.strip("() ")
                                    obj = obj.strip("() ")
                                    triples.append((subj, rel, obj))
                                except Exception as e:
                                    print(f"[WARN] Skipping malformed local triple: {line} ({e})")
                    else:
                        print(f"[WARN] No triples extracted for chunk {i+1}")
            except Exception as outer_e:
                print(f"[ERROR] Skipping chunk {i+1} due to extraction failure: {outer_e}")
                continue
            for subj, rel, obj in triples:
                cypher = f"""
                MERGE (a:Entity {{name: \"{subj}\"}})
                MERGE (b:Entity {{name: \"{obj}\"}})
                MERGE (a)-[:{rel.upper()}]->(b)
                """
                try:
                    graph.run_cypher(cypher)
                except Exception as e:
                    print(f"[ERROR] Cypher failed: {e}\nQuery: {cypher}")
        except Exception as outer_e:
            print(f"[ERROR] Failed on chunk {i+1}: {outer_e}")

def parse_triples(response):
    lines = response["text"].strip().split("\n")
    triples = []
    for line in lines:
        if ")-[" in line and "]->(" in line:
            try:
                subj, rest = line.split(")-[")
                rel, obj = rest.split("]->(")
                subj = subj.strip("() ")
                rel = rel.strip("() ")
                obj = obj.strip("() ")
                triples.append((subj, rel, obj))
            except Exception as e:
                print(f"[WARN] Skipping malformed triple: {line} ({e})")
    return triples

def extract_field(text, field_name):
    pattern = rf"{field_name}:\s*(.+?)(?=\n[A-Z][a-z]+:|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_patient_info(text):
    name_match = re.search(r"(?i)Player's Name:\s*(.*)", text)
    date_match = re.search(r"(?i)Date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    location_match = re.search(r"(?i)Location:\s*(.*)", text)

    return {
        "name": name_match.group(1).strip() if name_match else "Unknown",
        "date": date_match.group(1).strip() if date_match else "1900-01-01",
        "location": location_match.group(1).strip() if location_match else "Unknown",
        "diagnosis": extract_field(text, "Diagnosis"),
        "recommendations": extract_field(text, "Recommendations"),
        "history": extract_field(text, "History"),
        "physical_exam": extract_field(text, "Physical Examination")
    }

import subprocess

def call_fhir_builder(patient_info, page_text):
    patient_fields = {
        "name": patient_info["name"],
        "dob": patient_info.get("date", "1900-01-01"),
        "gender": "unknown"
    }

    condition_fields = {
        "diagnosis": patient_info.get("diagnosis", ""),
        "history": patient_info.get("history", ""),
        "recommendations": patient_info.get("recommendations", ""),
        "physical_exam": patient_info.get("physical_exam", "")
    }

    input_data = json.dumps({
        "patient": patient_fields,
        "condition": condition_fields
    }) + "\n" + page_text

    # Use the current Python interpreter to run the to_fhir script (works cross-platform)
    command = [sys.executable, "to_fhir.py"]

    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    stdout, stderr = proc.communicate(input=input_data)

    if proc.returncode != 0:
        raise RuntimeError(f"FHIR subprocess failed:\n{stderr}")

    return json.loads(stdout)

def extract_patient_identifier(filename: str) -> str:
    patterns = [
        r'(?i)patient[_\s-]*(\d+)',
        r'(\d{3,})',
        r'(?i)([a-zA-Z]+\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    base_name = os.path.splitext(filename)[0]
    return base_name.split('_')[0] if '_' in base_name else base_name

def get_pdf_files_by_patient(folder_path: str) -> Dict[str, List[str]]:
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    patient_groups = defaultdict(list)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        patient_id = extract_patient_identifier(filename)
        patient_groups[patient_id].append(pdf_path)
    
    return dict(patient_groups)

# ---------- GraphRAG Pipeline ----------
class GraphRAG:
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.graph = GraphInterface(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        self.vstore = VectorStore(patient_id)

    def ingest_pdfs(self, pdf_paths: List[str], log_fn=print):
        """
        Ingest multiple PDFs for a single patient.
        log_fn: optional callable for live log streaming (defaults to print)
        """
        log_fn(f"\n[INFO] Processing {len(pdf_paths)} PDFs for patient {self.patient_id}")
        
        all_docs = []
        all_documents = []
        all_conditions = []
        patient_info = None
        
        for i, pdf_path in enumerate(pdf_paths):
            pdf_name = os.path.basename(pdf_path)
            log_fn(f"\n{'='*60}")
            log_fn(f"Processing PDF {i+1}/{len(pdf_paths)}: {pdf_name}")
            log_fn(f"{'='*60}")
            
            processed_pdf_path = apply_targeted_ocr_with_ocrmypdf(pdf_path)
            docs = load_pdf_chunks_enhanced(processed_pdf_path)

            if not docs:
                log_fn(f"[ERROR] No content extracted from {pdf_name}. Skipping.")
                continue

            if patient_info is None or patient_info["name"] == "Unknown":
                patient_info = extract_patient_info(docs[0].page_content)
            
            all_docs.extend(docs)

            for doc in docs:
                try:
                    result = call_fhir_builder(patient_info, doc.page_content)
                    all_documents.append(result["document"])
                    all_conditions.append(result["condition"])
                except Exception as e:
                    log_fn(f"[ERROR] FHIR processing failed for {pdf_name}: {e}")

            try:
                summary_question = f"Give me a summary of the patient health or injury from {pdf_name}"
                summary = self.generate_summary(docs, summary_question)
                log_fn(f"\n--- Summary for {pdf_name} ---")
                log_fn(summary)
                log_fn(f"--- End Summary for {pdf_name} ---\n")
            except Exception as e:
                log_fn(f"[ERROR] Summary generation failed for {pdf_name}: {e}")

        if all_docs:
            log_fn(f"\n[INFO] Adding {len(all_docs)} total chunks to vector store for patient {self.patient_id}")
            self.vstore.add_documents(all_docs)
            
            log_fn(f"[INFO] Building knowledge graph for patient {self.patient_id}")
            extract_triples_and_populate_kg(all_docs, self.graph)

        if all_documents and patient_info:
            combined_output = {
                "patient": {
                    "id": self.patient_id,
                    "name": patient_info["name"],
                    "date": patient_info["date"],
                    "location": patient_info["location"]
                },
                "documents": all_documents,
                "conditions": all_conditions
            }

            os.makedirs("fhir", exist_ok=True)
            output_path = f"fhir/combined_{self.patient_id}.json"
            with open(output_path, "w") as f:
                json.dump(combined_output, f, indent=2)

            log_fn(f"\n[INFO] Saved combined FHIR JSON to {output_path}")
        
        return patient_info

    def generate_summary(self, docs: List[Document], question: str) -> str:
        if not docs:
            return "No documents available for summary."
        
        sample_docs = docs[:3] if len(docs) > 3 else docs
        vector_texts = [doc.page_content for doc in sample_docs]
        
        try:
            if 'answer_chain' in globals() and answer_chain is not None and USE_LLM:
                summary_result = safe_llm_call(answer_chain.invoke, {
                    "question": question,
                    "graph_results": "[]",
                    "vector_results": "\n".join(vector_texts)
                })
                return summary_result.get("text", "Summary generation failed.")
            elif USE_LOCAL_LLM:
                prompt_text = answer_prompt.format(question=question, graph_results="[]", vector_results="\n".join(vector_texts))
                response = local_llm_call(prompt_text, max_tokens=512)
                return response.get("text", "") or local_summarize(vector_texts, question)
            else:
                # Local summarizer fallback
                return local_summarize(vector_texts, question)
        except Exception as e:
            return f"Summary generation error: {e}"

    def run(self, user_input: str) -> str:
        parsed_input = QueryInput(question=user_input)
        # Generate cypher query (LLM if available, else simple fallback)
        if 'cypher_chain' in globals() and cypher_chain is not None and USE_LLM:
            result = safe_llm_call(cypher_chain.invoke, {"question": parsed_input.question})
            cypher_query = result.get("text", "").strip()
        elif USE_LOCAL_LLM:
            # create a simple cypher prompt and ask the local LLM
            prompt_text = cypher_prompt.format(question=parsed_input.question)
            response = local_llm_call(prompt_text, max_tokens=256)
            cypher_query = response.get("text", "").strip() or "MATCH (n) RETURN n LIMIT 10"
        else:
            cypher_query = "MATCH (n) RETURN n LIMIT 10"

        print(f"[DEBUG] Generated Cypher: {cypher_query}")

        try:
            graph_results = self.graph.run_cypher(cypher_query)
        except Exception as e:
            graph_results = [{"error": str(e)}]

        vector_results = self.vstore.search(parsed_input.question)
        vector_texts = [doc.page_content for doc in vector_results]

        if 'answer_chain' in globals() and answer_chain is not None and USE_LLM:
            final_answer = safe_llm_call(answer_chain.invoke, {
                "question": parsed_input.question,
                "graph_results": str(graph_results),
                "vector_results": "\n".join(vector_texts)
            })
            return final_answer.get("text", "Answer generation failed.")
        else:
            # Local deterministic answer combining simple graph info + vector text
            combined = "\n\n".join(vector_texts[:3])
            return f"[Local answer]\nQuestion: {parsed_input.question}\n\nGraph results: {graph_results}\n\nText snippets:\n{combined}"

# ---------- Main (CLI mode) ----------
if __name__ == "__main__":
    print("=== Multi-Patient PDF Processing System ===")
    
    patient_groups = get_pdf_files_by_patient(PDF_FOLDER)
    
    if not patient_groups:
        print(f"[ERROR] No PDF files found in {PDF_FOLDER}")
        exit()
    
    print(f"\n[INFO] Found {len(patient_groups)} patients with PDFs:")
    for patient_id, pdf_list in patient_groups.items():
        print(f"  - Patient {patient_id}: {len(pdf_list)} PDFs")
    
    for patient_id, pdf_paths in patient_groups.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING PATIENT: {patient_id}")
        print(f"{'='*80}")
        
        rag = GraphRAG(patient_id)
        rag.ingest_pdfs(pdf_paths)
        
        print(f"\n[INFO] Completed processing for patient {patient_id}")
        rag.graph.close()
    
    print(f"\n{'='*80}")
    print("ALL PATIENTS PROCESSED SUCCESSFULLY")
    print(f"{'='*80}")