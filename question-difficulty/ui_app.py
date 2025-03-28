import streamlit as st
import pandas as pd
import json
import csv
import io
import logging
import sys
import os
import time
try:
    from cache_manager import get_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

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

# --- Cache Controls (New Section) ---
with st.expander("Cache Settings", expanded=False):
    if CACHE_AVAILABLE:
        use_cache = st.checkbox("Enable Response Caching", value=True, 
                               help="Cache LLM responses to speed up repeated evaluations")
        cache_dir = st.text_input("Cache Directory", 
                                 value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"),
                                 help="Directory to store cached responses")
        ttl_days = st.slider("Cache TTL (days)", min_value=1, max_value=90, value=30,
                            help="Time-to-live for cached responses")
        
        # Initialize cache with the specified settings
        cache = get_cache(cache_dir=cache_dir, ttl_days=ttl_days, enabled=use_cache)
        
        # Add clear cache button
        if st.button("Clear Cache"):
            cleared = cache.clear_all()
            st.success(f"Cleared {cleared} cached responses")
        
        # Show cache statistics if available
        if cache.enabled:
            stats = cache.get_stats()
            st.write("Cache Statistics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hits", stats["hits"])
            with col2:
                st.metric("Misses", stats["misses"])
            with col3:
                hit_rate = stats["hit_rate"] if stats["total"] > 0 else 0
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
    else:
        st.warning("Caching is not available. Install the cache_manager module to enable caching.")
        use_cache = False

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
    # Add workers selection slider
    max_workers = st.slider("Max Workers", min_value=1, max_value=10, value=5,
                           help="Maximum number of parallel workers")
    
    if st.button("Compute Difficulty"):
        with st.spinner("Analyzing difficulty..."):
            # Track execution time
            start_time = time.time()
            
            # Pass the cache settings to the assessment function
            results = parallel_assess_difficulty(
                questions=st.session_state.passages,
                client=st.session_state.anthropic_client,
                max_workers=max_workers,
                use_cache=use_cache if CACHE_AVAILABLE else False
            )
            
            elapsed_time = time.time() - start_time
            st.session_state.analysis_results = results
            
            # Show execution time
            st.success(f"Analysis completed in {elapsed_time:.2f} seconds " +
                     f"({elapsed_time / len(st.session_state.passages):.2f} seconds per question)")
            
            # Show cache statistics after computation if caching is enabled
            if CACHE_AVAILABLE and use_cache:
                stats = cache.get_stats()
                st.info(f"Cache performance: {stats['hits']} hits, {stats['misses']} misses " +
                       f"({stats['hit_rate']:.1f}% hit rate)")

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
