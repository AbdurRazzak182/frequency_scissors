import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import logging_utils

st.set_page_config(page_title="History Log", page_icon="📜", layout="wide")
st.title("📜 Edit History")
st.caption(
    "Every frequency edit, noise removal, and mix performed in the "
    "Audio Studio is recorded here, newest first."
)

df = logging_utils.load_log()

if df.empty:
    st.info(
        "No actions logged yet. Head to the **Audio Studio** page and edit, "
        "denoise, or mix a clip — every step will show up here."
    )
    st.stop()

df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------
# 1. Summary
# ----------------------------------------------------------------------
st.subheader("1. Summary")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Total actions", len(df))
m_col2.metric("Clips touched", df["source_filename"].nunique())
m_col3.metric("Noise-removal passes", int((df["action_type"] == "noise_remove").sum()))
m_col4.metric("Mixes performed", int((df["action_type"] == "mix").sum()))

st.markdown("**Actions by Type**")
counts = df["action_type"].value_counts()
labels = [logging_utils.ACTION_LABELS.get(a, a) for a in counts.index]
bar_fig = go.Figure()
bar_fig.add_trace(go.Bar(
    x=counts.values, y=labels, orientation="h",
    marker=dict(color=["#2dd4bf", "#38bdf8", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399"][: len(labels)]),
))
bar_fig.update_layout(
    height=220, margin=dict(l=10, r=10, t=10, b=10),
    template="plotly_dark", xaxis_title="Count",
)
st.plotly_chart(bar_fig, use_container_width=True, key="history_summary_chart")

st.divider()

# ----------------------------------------------------------------------
# 2. Filter
# ----------------------------------------------------------------------
st.subheader("2. Filter")
filter_box = st.container(border=True)
with filter_box:
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        clip_options = ["All clips"] + sorted(df["source_filename"].dropna().unique().tolist())
        clip_filter = st.selectbox("Clip", options=clip_options)
    with f_col2:
        type_options = ["All actions"] + sorted(df["action_type"].dropna().unique().tolist())
        type_filter = st.selectbox(
            "Action type", options=type_options,
            format_func=lambda a: a if a == "All actions" else logging_utils.ACTION_LABELS.get(a, a),
        )

filtered = df.copy()
if clip_filter != "All clips":
    filtered = filtered[filtered["source_filename"] == clip_filter]
if type_filter != "All actions":
    filtered = filtered[filtered["action_type"] == type_filter]

st.divider()

# ----------------------------------------------------------------------
# 3. Timeline
# ----------------------------------------------------------------------
st.subheader(f"3. Timeline ({len(filtered)} entries)")

timeline_box = st.container(border=True)
with timeline_box:
    if filtered.empty:
        st.caption("No entries match this filter.")
    for _, row in filtered.iterrows():
        try:
            details = json.loads(row["details"]) if row["details"] else {}
        except (TypeError, json.JSONDecodeError):
            details = {}

        icon_label = logging_utils.ACTION_LABELS.get(row["action_type"], row["action_type"])
        description = logging_utils.describe_action(row["action_type"], details)

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{icon_label}** — {description}")
                st.caption(f"Clip: `{row['source_filename']}`")
            with c2:
                st.caption(row["timestamp"])
                if row["sample_rate"]:
                    st.caption(f"{int(row['sample_rate'])} Hz · {row['duration_sec']:.2f}s")
            if details:
                with st.expander("Raw parameters"):
                    st.json(details)

st.divider()

# ----------------------------------------------------------------------
# 4. Raw log table + export / clear
# ----------------------------------------------------------------------
st.subheader("4. Raw Log Table")
with st.container(border=True):
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    dl_col, clear_col = st.columns(2)
    with dl_col:
        st.download_button(
            "⬇️ Download full log (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="session_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with clear_col:
        if st.button("🗑️ Clear entire history", use_container_width=True):
            st.session_state["_confirm_clear_history"] = True

    if st.session_state.get("_confirm_clear_history"):
        st.warning("This permanently deletes the whole log. Are you sure?")
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("Yes, delete it", type="primary", use_container_width=True):
                logging_utils.clear_log()
                st.session_state["_confirm_clear_history"] = False
                st.rerun()
        with no_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state["_confirm_clear_history"] = False
                st.rerun()