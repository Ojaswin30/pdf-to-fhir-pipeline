import streamlit as st
import glob
import json
import os
import shutil
import tempfile

st.set_page_config(page_title="PDF → FHIR Explorer", layout="wide")

current_nav = st.query_params.get("nav", "home")
if isinstance(current_nav, list):
    current_nav = current_nav[0] if current_nav else "home"
nav = str(current_nav).lower()
if nav not in {"home", "records"}:
    nav = "home"

with st.sidebar:
    st.page_link("https://ojaswin30.github.io/", label="Back", icon="⬅️")

    st.markdown("### Navigation")

    if st.button("Home", use_container_width=True):
        st.query_params["nav"] = "home"
        st.rerun()

    if st.button("Records", use_container_width=True):
        st.query_params["nav"] = "records"
        st.rerun()

st.title("PDF → FHIR Explorer")

fhir_files = sorted(glob.glob(os.path.join("fhir", "combined_*.json")))

if nav == "home":
    st.header("Upload & Process PDF")
    st.write("Upload a PDF and choose whether to only extract text or run the full pipeline.")

    upload = st.file_uploader("Upload a PDF to process", type=["pdf"], key="home_upload")
    process_mode = st.selectbox("Processing mode", ["Extract only", "Run full pipeline (may require keys)"])

    if upload:
        uploads_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        uploaded_path = os.path.join(uploads_dir, upload.name)
        with open(uploaded_path, "wb") as f:
            f.write(upload.getbuffer())

        st.success(f"Saved to {uploaded_path}")

        # Lightweight extraction using Main's utilities if available
        try:
            from Main import get_comprehensive_text_from_pdf, load_pdf_chunks_enhanced, GraphRAG

            st.subheader("Extracted Text (first 2000 chars)")
            with st.spinner("Extracting text..."):
                text = get_comprehensive_text_from_pdf(uploaded_path)
            st.text(text[:2000] + ("..." if len(text) > 2000 else ""))

            st.subheader("Chunks")
            with st.spinner("Splitting into chunks..."):
                chunks = load_pdf_chunks_enhanced(uploaded_path)
            st.write(f"{len(chunks)} chunks")
            if chunks:
                for i, c in enumerate(chunks[:3], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(c.page_content[:1000])

            if process_mode == "Run full pipeline (may require keys)":
                if st.button("Run full pipeline on this PDF"):
                    st.info("Running GraphRAG.ingest_pdfs — this may call external LLMs and require API keys.")
                    try:
                        rag = GraphRAG(os.path.splitext(upload.name)[0])
                        with st.spinner("Processing PDF (this may take a while)..."):
                            rag.ingest_pdfs([uploaded_path], log_fn=lambda m: st.write(m))
                        st.success("Processing finished. Check fhir/ and vector_store/ for outputs.")
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")

        except Exception as e:
            st.error(f"Could not run extraction/pipeline utilities: {e}")

elif nav == "records":
    st.header("FHIR Records")
    if not fhir_files:
        st.warning("No FHIR JSON files found in the `fhir/` folder.")
    else:
        patients = [os.path.basename(p).replace("combined_", "").replace(".json", "") for p in fhir_files]
        sel = st.selectbox("Select patient record", patients)
        selected_file = os.path.join("fhir", f"combined_{sel}.json")
        with open(selected_file, "r") as f:
            data = json.load(f)

        st.subheader(f"Patient: {data.get('patient', {}).get('name', sel)}")
        # Allow selecting a particular PDF (if available in documents)
        docs = data.get("documents", [])
        pdf_choices = [f"Document {i+1}" for i in range(len(docs))] if docs else []
        chosen = st.selectbox("Select document to view", ["Full record"] + pdf_choices)

        cols = st.columns([2, 1])

        with cols[0]:
            st.subheader("FHIR JSON")
            st.json(data)

        with cols[1]:
            st.subheader("Summary / Documents")
            if chosen == "Full record":
                st.text(json.dumps(data.get('patient', {}), indent=2))
            else:
                idx = pdf_choices.index(chosen)
                st.text(docs[idx] if isinstance(docs[idx], str) else json.dumps(docs[idx], indent=2))

        st.markdown("---")
        st.caption("This UI shows combined FHIR JSON files produced by the pipeline.")
