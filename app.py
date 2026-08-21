import streamlit as st


st.set_page_config(
    page_title="Frequency Scissors",
    page_icon="✂️",
    layout="wide",
)

st.title("✂️ Frequency Scissors")

st.markdown(
    ":violet-badge[:material/star: An interactive frequency-domain audio editing playground] "
)


st.header("🎯 About This Project")
st.write(
    """
    **Frequency Scissors** is an audio-processing tool that lets you
    upload or record a short audio clip, visualize its **waveform**
    and **frequency spectrum**, and interactively "cut" selected
    frequency ranges using draggable frequency-band scissors.

    The modified audio is reconstructed using the **inverse Fourier
    transform (IFFT)**, so you can immediately hear how removing,
    attenuating, or amplifying different frequency bands changes the
    sound. The original and processed audio are compared side by
    side — waveform plots, spectra, and listening tests — to
    demonstrate practical frequency-domain filtering and audio
    editing.

    Under the hood, every clip is kept as a raw **NumPy** array in
    memory, and a **Pandas** DataFrame logs metadata for every clip
    and every set of cuts you apply (sample rate, duration, and the
    history of frequency bands you've cut).
    """
)

st.header("📁 Project Progression")
st.info(
    "👉**Phase 1 · Data Ingestion & Time-Domain Visualization."
)

st.info(
    "❌**Phase 2 ·The Mathematical Engine (STFT)."
)
st.info(
    "❌ **Phase 3 · The Interactive Scissors (UI)."
)
st.info(
    "❌ **Phase 4 · Reconstruction & Artifact Management."
)
st.info(
    "❌ **Phase 5 · Testing & Output."
)

st.header("👋 About Developers")
dev1,dev2 = st.columns(2)
with dev1:
    st.markdown("""
    <style>
        [data-testid="stImage"] img {
            border-radius: 50%;
            object-fit: cover;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.15);
        }
    </style>
    """, unsafe_allow_html=True)
    st.image("https://github.com/AbdurRazzak182.png", width=150)
    st.subheader("Abdur Razzak")
    st.caption("2305110")
    st.caption("Undergraduate in CSE")
    st.markdown("[GitHub](https://github.com/AbdurRazzak182) | [LinkedIn](https://https://www.linkedin.com/feed/)")

with dev2:
    st.image("https://api.dicebear.com/7.x/avataaars/svg?seed=dipto", width=150)
    st.subheader("Dipto Debnath")
    st.caption("2305111")
    st.caption("Undergraduate in CSE")
    st.markdown("[GitHub](https://) | [LinkedIn](https://https:///)")



