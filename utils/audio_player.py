import base64
import json
import mimetypes
import streamlit.components.v1 as components


def render_audio_player(audio_bytes: bytes, filename: str, peaks: list,
                         key: str, height: int = 170):
    mime = mimetypes.guess_type(filename)[0] or "audio/wav"
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    peaks_json = json.dumps(peaks)

    html = f"""
    <div style="font-family: sans-serif;">
      <canvas id="wf_{key}" width="900" height="110"
              style="width:100%; height:110px; background:#0f172a;
                     border-radius:10px; display:block;"></canvas>
      <audio id="audio_{key}" controls style="width:100%; margin-top:10px;">
        <source src="data:{mime};base64,{b64}" type="{mime}">
        Your browser does not support the audio element.
      </audio>
    </div>
    <script>
    (function() {{
        const peaks = {peaks_json};
        const canvas = document.getElementById("wf_{key}");
        const ctx = canvas.getContext("2d");
        const audio = document.getElementById("audio_{key}");

        function draw(progress) {{
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            const barW = w / peaks.length;
            for (let i = 0; i < peaks.length; i++) {{
                const amp = Math.max(peaks[i] * (h / 2 - 4), 1);
                const playedColor = (i / peaks.length) <= progress;
                ctx.fillStyle = playedColor ? "#2dd4bf" : "#475569";
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

        draw(0);

        audio.addEventListener("timeupdate", function() {{
            const progress = audio.duration ? audio.currentTime / audio.duration : 0;
            draw(progress);
        }});
        audio.addEventListener("seeked", function() {{
            const progress = audio.duration ? audio.currentTime / audio.duration : 0;
            draw(progress);
        }});
        audio.addEventListener("ended", function() {{ draw(0); }});

        // click-to-seek on the canvas
        canvas.addEventListener("click", function(evt) {{
            const rect = canvas.getBoundingClientRect();
            const x = evt.clientX - rect.left;
            const frac = x / rect.width;
            if (audio.duration) {{
                audio.currentTime = frac * audio.duration;
            }}
        }});
    }})();
    </script>
    """
    components.html(html, height=height + 40)