import streamlit as st


st.set_page_config(
    page_title="Frequency Scissors",
    page_icon="✂️",
    layout="wide",
)

st.title("Audio Frequency Processing")

st.markdown(
    ":violet-badge[:material/star: An interactive frequency-domain audio editing playground] "
)


st.header("🎯 About This Project")
st.write(
    """
    **Audio Frequency Processing is a browser-based audio editing workstation built with Streamlit that lets you manipulate sound in the frequency domain. You can upload an audio clip, visualize its waveform and frequency spectrum, and remove, attenuate, amplify, or isolate specific frequency bands using FFT-based reconstruction. The project also includes a full noise-handling pipeline — you can inject stationary, drifting, or burst/click noise for testing, then detect and clean it out using STFT-based spectral subtraction and transient detection. A multi-track mixer lets you combine several clips with individual volume control, and every edit is tracked as a version with an automatic before/after comparison panel so changes can be verified visually and audibly. A session-wide history log records every action taken. Altogether, it works as a lightweight, interactive tool for exploring and demonstrating frequency-domain audio processing concepts.
    """
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



