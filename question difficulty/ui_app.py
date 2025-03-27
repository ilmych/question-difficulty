import streamlit as st
import pandas as pd
import json
import csv
import io
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from difficulty_framework import generate_difficulty_report
from parallel_implementation import parallel_assess_difficulty
import anthropic

# ---------------- Setup ----------------
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Question Difficulty Framework", layout="wide")
st.title("Question Difficulty Framework")

# --- Session State Setup ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "passages" not in st.session_state:
    st.session_state.passages = None
if "anthropic_client" not in st.session_state:
    st.session_state.anthropic_client = None

# ---------------- API Key ----------------
api_key_input = st.text_input("Enter your Anthropic API key", type="password")

if api_key_input and st.session_state.anthropic_client is None:
    st.session_state.anthropic_client = anthropic.Anthropic(api_key=api_key_input)
    st.success("Anthropic client initialized.")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your input file (.json or .csv)", type=["json", "csv"])

if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")

    if uploaded_file.name.endswith(".json"):
        st.session_state.passages = json.loads(file_contents)
    else:
        reader = csv.DictReader(io.StringIO(file_contents))
        st.session_state.passages = list(reader)

    st.write(f"✅ Loaded {len(st.session_state.passages)} passages/questions from **{uploaded_file.name}**")

# ---------------- Compute Difficulty ----------------
if st.session_state.passages is not None:
    if st.button("Compute Difficulty"):
        with st.spinner("Analyzing difficulty..."):
            results = parallel_assess_difficulty(
                questions=st.session_state.passages,
                client=st.session_state.anthropic_client
            )
            st.session_state.analysis_results = results

# ---------------- Display Results ----------------
if st.session_state.analysis_results is not None:
    results = st.session_state.analysis_results
    question_data = results["questions"]

    # --- Overall Summary ---
    overall = results["overall_test_difficulty"]
    st.subheader("📊 Overall Test Difficulty")
    st.write("**Level:**", overall["level"])
    st.write("**Average Score:**", f"{overall['score']:.2f}")

    # ---------------- Generate Report ----------------
    st.subheader("📝 Generate Difficulty Report")
    report_format = st.selectbox("Select report format", ["html", "markdown", "text"])

    if st.button("Generate Report"):
        report = generate_difficulty_report(
            question_results=question_data,
            overall_result=overall,
            output_format=report_format,
            output_file=None
        )

        # Choose correct MIME type
        mime_type = {
            "html": "text/html",
            "markdown": "text/markdown",
            "text": "text/plain"
        }[report_format]

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"difficulty_report.{report_format}",
            mime=mime_type
        )

        if report_format == "html":
            st.markdown(report, unsafe_allow_html=True)
        elif report_format == "markdown":
            st.markdown(report)
        else:
            st.text(report)
