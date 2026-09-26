import base64
import json
import mimetypes 
import streamlit.components.v1 as components


def render_audio_player(audio_bytes: bytes, filename: str, peaks: list,
                         key: str, height: int = 210):
    mime = mimetypes.guess_type(filename)[0] or "audio/wav"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    peaks_json = json.dumps(peaks)

    html = f"""
    <div class="apc-wrap">
      <style>
        .apc-wrap {{ font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }}
        .apc-canvas-box {{
          background:#0f172a; border-radius:12px; padding:2px;
          border:1px solid rgba(148,163,184,0.18);
        }}
        #apc_wf_{key} {{ width:100%; height:96px; display:block; cursor:pointer; border-radius:10px; }}
        .apc-controls {{
          display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap;
        }}
        .apc-btn {{
          border:none; cursor:pointer; display:flex; align-items:center; justify-content:center;
          font-weight:700; font-size:15px; transition:transform .12s ease, box-shadow .12s ease;
        }}
        .apc-btn:active {{ transform:scale(0.94); }}
        .apc-btn-round {{
          width:38px; height:38px; border-radius:50%;
          background:rgba(148,163,184,0.14); color:#e2e8f0;
          border:1px solid rgba(148,163,184,0.25);
        }}
        .apc-btn-round:hover {{ background:rgba(148,163,184,0.26); }}
        .apc-btn-play {{
          width:52px; height:52px; border-radius:50%;
          background:linear-gradient(135deg,#2dd4bf,#38bdf8);
          color:#04121a; font-size:20px;
          box-shadow:0 0 0 rgba(56,189,248,0.0);
        }}
        .apc-btn-play:hover {{ box-shadow:0 0 18px rgba(56,189,248,0.55); }}
        .apc-time {{
          color:#94a3b8; font-size:13px; font-variant-numeric:tabular-nums;
          min-width:96px;
        }}
        .apc-time b {{ color:#e2e8f0; }}
        .apc-vol {{
          display:flex; align-items:center; gap:6px; margin-left:auto; color:#94a3b8; font-size:15px;
        }}
        .apc-vol input[type=range] {{
          width:80px; accent-color:#38bdf8; cursor:pointer;
        }}
      </style>

      <div class="apc-canvas-box">
        <canvas id="apc_wf_{key}" width="900" height="96"></canvas>
      </div>

      <div class="apc-controls">
        <button id="apc_restart_{key}" class="apc-btn apc-btn-round" title="Restart from 0:00">⏮</button>
        <button id="apc_toggle_{key}" class="apc-btn apc-btn-play" title="Play / Pause">▶</button>
        <button id="apc_stop_{key}" class="apc-btn apc-btn-round" title="Stop">⏹</button>
        <div class="apc-time"><b id="apc_pos_{key}">0:00</b> / <span id="apc_dur_{key}">0:00</span></div>
        <div class="apc-vol">
          🔊<input type="range" id="apc_vol_{key}" min="0" max="1" step="0.01" value="1">
        </div>
      </div>

      <audio id="apc_audio_{key}" style="display:none;" preload="metadata">
        <source src="data:{mime};base64,{b64}" type="{mime}">
      </audio>
    </div>
    <script>
    (function() {{
        const peaks = {peaks_json};
        const canvas = document.getElementById("apc_wf_{key}");
        const ctx = canvas.getContext("2d");
        const audio = document.getElementById("apc_audio_{key}");
        const posLabel = document.getElementById("apc_pos_{key}");
        const durLabel = document.getElementById("apc_dur_{key}");
        const toggleBtn = document.getElementById("apc_toggle_{key}");
        const restartBtn = document.getElementById("apc_restart_{key}");
        const stopBtn = document.getElementById("apc_stop_{key}");
        const volSlider = document.getElementById("apc_vol_{key}");

        function fmtTime(sec) {{
            if (!isFinite(sec) || sec < 0) sec = 0;
            const m = Math.floor(sec / 60);
            const s = Math.floor(sec % 60).toString().padStart(2, "0");
            return m + ":" + s;
        }}

        function draw(progress) {{
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            const barW = w / peaks.length;
            for (let i = 0; i < peaks.length; i++) {{
                const amp = Math.max(peaks[i] * (h / 2 - 4), 1);
                const played = (i / peaks.length) <= progress;
                ctx.fillStyle = played ? "#2dd4bf" : "#334155";
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
            posLabel.textContent = fmtTime(audio.currentTime);
            const progress = audio.duration ? audio.currentTime / audio.duration : 0;
            draw(progress);
        }}

        function updateToggleIcon() {{
            toggleBtn.innerHTML = audio.paused ? "▶" : "⏸";
        }}

        audio.addEventListener("loadedmetadata", function() {{
            durLabel.textContent = fmtTime(audio.duration);
        }});
        audio.addEventListener("timeupdate", updatePosition);
        audio.addEventListener("seeked", updatePosition);
        audio.addEventListener("ended", updateToggleIcon);
        audio.addEventListener("play", updateToggleIcon);
        audio.addEventListener("pause", updateToggleIcon);

        // click / drag-to-seek on the waveform
        canvas.addEventListener("click", function(evt) {{
            const rect = canvas.getBoundingClientRect();
            const frac = (evt.clientX - rect.left) / rect.width;
            if (audio.duration) {{
                audio.currentTime = frac * audio.duration;
            }}
        }});

        // --- Controls ---
        restartBtn.addEventListener("click", function() {{
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

        stopBtn.addEventListener("click", function() {{
            audio.pause();
            audio.currentTime = 0;
            updatePosition();
        }});

        volSlider.addEventListener("input", function() {{
            audio.volume = parseFloat(volSlider.value);
        }});

        draw(0);
        updateToggleIcon();
    }})();
    </script>
    """
    components.html(html, height=height)