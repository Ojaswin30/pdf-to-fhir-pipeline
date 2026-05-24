import streamlit as st
import glob
import json
import os
import shutil
import tempfile

st.set_page_config(page_title="PDF → FHIR Explorer", layout="wide")

st.title("PDF → FHIR Explorer")

fhir_files = sorted(glob.glob(os.path.join("fhir", "combined_*.json")))

# Sidebar navigation
nav = st.sidebar.radio("Navigation", ["Home", "Records"], index=0)

if nav == "Home":
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
                        import streamlit as st
                        import glob
                        import json
                        import os
                        import shutil
                        import tempfile

                        st.set_page_config(page_title="PDF → FHIR Explorer", layout="wide")

                        # Simple CSS to make the top navbar and cards look nicer
                        st.markdown(
                            """
                            <style>
                            .header {display:flex;align-items:center;gap:16px}
                            .logo {font-weight:700;font-size:20px;color:#0b5fff}
                            .subtle {color: #6b7280}
                            .metric {background:#f8fafc;padding:12px;border-radius:8px}
                            .card {padding:12px;border-radius:8px;background:linear-gradient(180deg, #ffffff, #fbfdff);box-shadow:0 1px 2px rgba(16,24,40,0.04)}
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        header_cols = st.columns([0.5, 3, 1])
                        with header_cols[0]:
                            st.image("", width=48)
                        with header_cols[1]:
                            st.markdown('<div class="header"><div class="logo">PDF → FHIR Pipeline</div><div class="subtle">Lightweight, local-first processing and FHIR export</div></div>', unsafe_allow_html=True)
                        with header_cols[2]:
                            st.write("")

                        # Top navigation tabs
                        tabs = st.tabs(["Home", "Records"])

                        # Basic metrics
                        pdf_count = len(glob.glob(os.path.join("pdf", "*.pdf")))
                        fhir_files = sorted(glob.glob(os.path.join("fhir", "combined_*.json")))
                        fhir_count = len(fhir_files)
                        vector_count = len(glob.glob(os.path.join("vector_store", "faiss_index_*.pkl")))

                        with tabs[0]:
                            st.subheader("Upload & Process PDF")
                            col1, col2 = st.columns([2, 1])
                            with col2:
                                st.markdown("<div class='card'>", unsafe_allow_html=True)
                                st.markdown(f"**PDFs:** {pdf_count}<br>**FHIR records:** {fhir_count}<br>**Vector indexes:** {vector_count}", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                            uploaded_file = st.file_uploader("Upload a PDF to process", type=["pdf"], key="home_upload")
                            process_mode = st.radio("Processing mode", ["Extract only", "Run full pipeline (may require keys)"], horizontal=True)

                            if uploaded_file:
                                uploads_dir = os.path.join(os.getcwd(), "uploads")
                                os.makedirs(uploads_dir, exist_ok=True)
                                uploaded_path = os.path.join(uploads_dir, uploaded_file.name)
                                with open(uploaded_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())

                                st.success(f"Saved to {uploaded_path}")

                                try:
                                    from Main import get_comprehensive_text_from_pdf, load_pdf_chunks_enhanced, GraphRAG

                                    with st.expander("Extraction Preview", expanded=True):
                                        with st.spinner("Extracting text..."):
                                            text = get_comprehensive_text_from_pdf(uploaded_path)
                                        st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

                                    with st.expander("Chunks", expanded=False):
                                        with st.spinner("Splitting into chunks..."):
                                            chunks = load_pdf_chunks_enhanced(uploaded_path)
                                        st.write(f"{len(chunks)} chunks")
                                        for i, c in enumerate(chunks[:5], 1):
                                            st.markdown(f"**Chunk {i}**")
                                            st.text(c.page_content[:1000])

                                    if process_mode == "Run full pipeline (may require keys)":
                                        if st.button("Run full pipeline on this PDF"):
                                            st.info("Processing — this may call local or remote LLMs and can take a while.")
                                            try:
                                                rag = GraphRAG(os.path.splitext(uploaded_file.name)[0])
                                                with st.spinner("Processing PDF..."):
                                                    rag.ingest_pdfs([uploaded_path], log_fn=lambda m: st.write(m))
                                                st.success("Processing finished. Check FHIR Records and vector_store for outputs.")
                                            except Exception as e:
                                                st.error(f"Pipeline failed: {e}")

                                except Exception as e:
                                    st.error(f"Could not run extraction/pipeline utilities: {e}")

                        with tabs[1]:
                            st.subheader("FHIR Records")
                            if not fhir_files:
                                st.warning("No FHIR JSON files found in the `fhir/` folder.")
                            else:
                                patients = [os.path.basename(p).replace("combined_", "").replace(".json", "") for p in fhir_files]
                                sel = st.selectbox("Select patient record", patients)
                                selected_file = os.path.join("fhir", f"combined_{sel}.json")
                                with open(selected_file, "r") as f:
                                    data = json.load(f)

                                top_cols = st.columns([3, 1])
                                with top_cols[0]:
                                    st.markdown(f"### {data.get('patient', {}).get('name', sel)}")
                                    st.write(f"ID: {sel}")
                                    st.write(f"Date: {data.get('patient', {}).get('date', 'N/A')}")
                                    st.write(f"Documents: {len(data.get('documents', []))}")
                                with top_cols[1]:
                                    st.download_button("Download JSON", data=json.dumps(data, indent=2), file_name=f"combined_{sel}.json", mime="application/json")

                                st.markdown("---")
                                doc_list = data.get('documents', [])
                                if doc_list:
                                    chosen_idx = st.selectbox('View document', list(range(len(doc_list))), format_func=lambda i: f"Document {i+1}")
                                    st.subheader(f"Document {chosen_idx+1}")
                                    st.text(doc_list[chosen_idx] if isinstance(doc_list[chosen_idx], str) else json.dumps(doc_list[chosen_idx], indent=2))
                                else:
                                    st.info('No documents available for this patient.')

                            st.markdown("---")
                            st.caption("This UI shows combined FHIR JSON files produced by the pipeline. Use the Home tab to upload and process PDFs.")
