import streamlit as st
import tempfile
import os

from src.contradiction_detector import ContradictionDetector
from src.llm_explainer import LLMExplainer

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Legal Contract Contradiction Detection",
    page_icon="⚖️",
    layout="wide"
)

# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("⚖️ Legal Contract Contradiction Detection System")

st.write(
    "Upload one or more legal contract PDF files. The system automatically analyzes the contracts and identifies potential contradictions between related clauses."
)

uploaded_files = st.file_uploader(
    "Upload One or More PDF Contracts",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) uploaded successfully.")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")

# --------------------------------------------------
# Detect Button
# --------------------------------------------------

if st.button("🚀 Detect Contradictions", use_container_width=True):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    MODEL_PATH = "models/deberta_contractnli_best"

    detector = ContradictionDetector(model_path=MODEL_PATH)
    explainer = LLMExplainer()

    pdf_paths = []

    with tempfile.TemporaryDirectory() as temp_dir:

        # Save uploaded PDFs
        for uploaded_file in uploaded_files:

            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_paths.append(file_path)

        # Progress bar and status
        progress = st.progress(0)
        status = st.empty()

        status.info("📄 Extracting and processing documents...")
        detector.process_documents(pdf_paths)
        progress.progress(33)

        status.info("🧠 Building semantic search index...")
        detector.build_vector_index()
        progress.progress(66)

        status.info("⚖️ Detecting contradictions...")
        contradictions = detector.detect_contradictions(
            top_k=5,
            contradiction_threshold=0.70
        )
        progress.progress(100)

        status.success("✅ Analysis completed successfully.")

        st.success(
            f"Analysis completed successfully. "
            f"{len(contradictions)} contradiction(s) detected."
        )

        st.divider()

        st.header("📋 Detection Results")

        if len(contradictions) == 0:

            st.info("No contradictions were detected.")

        else:

            for i, c in enumerate(contradictions, start=1):

                with st.container(border=True):

                    st.subheader(f"Contradiction {i}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Prediction", c["label"])

                    with col2:
                        st.metric(
                            "Confidence",
                            f"{c['confidence']*100:.2f}%"
                        )

                    st.write(f"**Source Document:** {c['source_document']}")
                    st.write(f"**Target Document:** {c['target_document']}")
                    st.write(f"**Semantic Similarity:** {c['similarity']:.4f}")

                    with st.expander("📄 Source Clause"):
                        st.write(c["source_text"])

                    with st.expander("📄 Target Clause"):
                        st.write(c["target_text"])

                    explanation = explainer.explain(
                        source_text=c["source_text"],
                        target_text=c["target_text"]
                    )

                    with st.expander("🤖 AI Explanation"):
                        st.markdown(explanation)