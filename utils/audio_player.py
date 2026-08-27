import base64
import json
import mimetypes
import streamlit.components.v1 as components


def render_audio_player(audio_bytes: bytes, filename: str, peaks: list,
                         key: str, height: int = 260):
    mime = mimetypes.guess_type(filename)[0] or "audio/wav"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    peaks_json = json.dumps(peaks)

    html = f"""
    <div style="font-family: sans-serif;">

      <canvas id="wf_{key}" width="900" height="110"
              style="width:100%; height:110px; background:#0f172a;
                     border-radius:10px; display:block;"></canvas>

      <!-- Control bar (below waveform) -->
      <div style="display:flex; align-items:center; gap:10px; margin-top:12px;">
        <button id="btn_original_{key}"
                style="background:#2563eb; color:white; border:none;
                       padding:8px 18px; border-radius:20px; font-weight:600;
                       cursor:pointer; font-size:14px;">
          ▶ Play
        </button>
        <button id="btn_toggle_{key}"
                style="background:white; color:#1e293b; border:1px solid #cbd5e1;
                       padding:8px 18px; border-radius:20px; font-weight:600;
                       cursor:pointer; font-size:14px;">
          ▶ Resume
        </button>
        
        <span style="color:#64748b; font-size:14px; margin-left:6px;">
          Position : <b id="pos_{key}" style="color:green;"> 0.00 s</b>
        </span>
      </div>

      <audio id="audio_{key}" style="display:none;">
        <source src="data:{mime};base64,{b64}" type="{mime}">
      </audio>
    </div>
    <script>
    (function() {{
        const peaks = {peaks_json};
        const canvas = document.getElementById("wf_{key}");
        const ctx = canvas.getContext("2d");
        const audio = document.getElementById("audio_{key}");
        const posLabel = document.getElementById("pos_{key}");
        const toggleBtn = document.getElementById("btn_toggle_{key}");

        function draw(progress) {{
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            const barW = w / peaks.length;
            for (let i = 0; i < peaks.length; i++) {{
                const amp = Math.max(peaks[i] * (h / 2 - 4), 1);
                const played = (i / peaks.length) <= progress;
                ctx.fillStyle = played ? "#2dd4bf" : "#475569";
                ctx.fillRect(i * barW, h / 2 - amp, Math.max(barW - 1, 1), amp * 2);
            }}
            const x = progress * w;
            ctx.strokeStyle = "#f43f5e";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }}

        function updatePosition() {{
            posLabel.textContent = audio.currentTime.toFixed(2) + " s";
            const progress = audio.duration ? audio.currentTime / audio.duration : 0;
            draw(progress);
        }}

        function updateToggleLabel() {{
            toggleBtn.innerHTML = audio.paused ? "▶ Resume" : "⏸ Pause";
        }}

        draw(0);
        updateToggleLabel();

        audio.addEventListener("timeupdate", updatePosition);
        audio.addEventListener("seeked", updatePosition);
        audio.addEventListener("ended", updateToggleLabel);
        audio.addEventListener("play", updateToggleLabel);
        audio.addEventListener("pause", updateToggleLabel);

        // click-to-seek on the canvas
        canvas.addEventListener("click", function(evt) {{
            const rect = canvas.getBoundingClientRect();
            const frac = (evt.clientX - rect.left) / rect.width;
            if (audio.duration) {{
                audio.currentTime = frac * audio.duration;
            }}
        }});

        // --- Controls ---
        document.getElementById("btn_original_{key}").addEventListener("click", function() {{
            audio.currentTime = 0;
            audio.play();
        }});

        toggleBtn.addEventListener("click", function() {{
            if (audio.paused) {{
                audio.play();
            }} else {{
                audio.pause();
            }}
        }});

        document.getElementById("btn_stop_{key}").addEventListener("click", function() {{
            audio.pause();
            updatePosition();
        }});
    }})();
    </script>
    """
    components.html(html, height=height)