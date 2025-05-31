

from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket
import whisper
import tempfile
import shutil
from gtts import gTTS
import os
from pydantic import BaseModel
from pytube import YouTube
from starlette.responses import JSONResponse
from googletrans import Translator
import traceback
import yt_dlp
import io
import ffmpeg
from pydub import AudioSegment
from zipfile import ZipFile

from starlette.websockets import WebSocketDisconnect




# uvicorn main:app --host 0.0.0.0 --port 8000 --limit-concurrency 100 --max-upload-size 3145728000  # 3GB
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base" , device="cpu")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.post("/audio-dubbing")
async def audio_dubbing(
    audio: UploadFile = File(...),
    sourceLang: str = Form(...),
    targetLang: str = Form(...),
    title: str = Form(...)
):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", mode='wb') as tmp_audio:
            shutil.copyfileobj(audio.file, tmp_audio)
            audio_path = tmp_audio.name

        # Transcribe and translate
        result = model.transcribe(audio_path, language=targetLang, task="translate")
        translated_text = result["text"]

        # Convert to speech
        tts = gTTS(text=translated_text, lang=targetLang)
        tts_path = tempfile.mktemp(suffix=".mp3")
        tts.save(tts_path)

        # Clean up uploaded audio
        os.remove(audio_path)

        # Send TTS result back
        return FileResponse(tts_path, media_type="audio/mpeg", filename=f"{title}.mp3")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio dubbing failed: {e}")






#     ----------------------------------------
@app.post("/text-to-speech")
async def text_to_speech(text: str, lang: str = "en"):
    tts = gTTS(text=text, lang=lang)
    output_path = "output.mp3"
    tts.save(output_path)
    return FileResponse(output_path, media_type="audio/mpeg")



# Pydantic model for JSON input
class TranslationRequest(BaseModel):
    url: str
    source_language: str = 'en'
    target_language: str

# Download audio using yt-dlp
def download_audio(url):
    # Temporary audio file
    temp_audio_path = tempfile.mktemp(suffix=".mp3")
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioquality': 1,
        'outtmpl': temp_audio_path
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None
    return temp_audio_path





# ------------------------------------------------------------------------------------------
@app.post("/download-subtitles/")
async def download_subtitles(request: TranslationRequest):
    temp_audio_path = None
    output_audio_path = None

    try:
        # 🔹 1. Download YouTube audio using yt-dlp
        temp_audio_path = download_audio(request.url)
        if not temp_audio_path:
            return JSONResponse(status_code=500, content={"error": "Failed to download audio from YouTube."})

        # 🔹 2. Transcribe with Whisper
        try:
            result = model.transcribe(temp_audio_path, language=request.target_language)
            transcribed_text = result["text"]
            if not transcribed_text.strip():
                raise Exception("Transcription returned empty text.")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Whisper transcription failed: {str(e)}"})



        # 🔹 4. Text-to-Speech
        try:
            output_audio_path = tempfile.mktemp(suffix=".mp3")
            tts = gTTS(text=transcribed_text)
            tts.save(output_audio_path)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"TTS failed: {str(e)}"})

        # 🔹 5. Return audio file
        return FileResponse(output_audio_path, media_type="audio/mpeg", filename="dubbed_audio.mp3")

    except Exception as e:
        print("🔥 Unhandled exception:")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # 🔹 6. Cleanup
        try:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except Exception:
            pass









# -----------------------------------------------------------------------------

class ContactForm(BaseModel):
    name: str
    email: str
    message: str

@app.post("/contact")
async def contact(form: ContactForm):
    print("📨 New Contact Message:", form.dict())
    return {"success": True, "message": "Message received successfully!"}





# ------------------------------------------------------------------

translator = Translator()

@app.websocket("/ws/dub")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            print("🔊 Received audio chunk")

            # Save raw audio temporarily
            raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")  # assume it's webm/mp3/raw
            raw_tmp.write(audio_bytes)
            raw_tmp.close()

            # Convert to valid .wav using ffmpeg
            wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav_tmp.close()

            try:
                ffmpeg.input(raw_tmp.name).output(wav_tmp.name, format='wav').run(quiet=True, overwrite_output=True)

                result = model.transcribe(wav_tmp.name, language="en", fp16=False)
                transcription = result.get("text", "").strip()
                print("📝 Transcription:", transcription)

                if not transcription:
                    print("⚠️ Empty transcription")
                    continue

                # Translate using gTTS (e.g., to Hindi)
                tts = gTTS(text=transcription, lang='hi')
                tts_fp = io.BytesIO()
                tts.write_to_fp(tts_fp)

                await websocket.send_bytes(tts_fp.getvalue())

            except Exception as e:
                print("❌ Error during processing:", e)

            finally:
                os.unlink(raw_tmp.name)
                os.unlink(wav_tmp.name)

    except Exception as e:
        print("❌ Error:", e)

    finally:
        await websocket.close()
        print("🔌 WebSocket closed")




# ----------------------------------------------------------------------------

# Helper function to convert segments to .srt subtitles
def segments_to_srt(segments):
    def format_time(t):
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = t % 60
        return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')

    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        srt_lines.append(str(i))
        srt_lines.append(f"{format_time(seg['start'])} --> {format_time(seg['end'])}")
        srt_lines.append(seg['text'].strip())
        srt_lines.append("")

    return "\n".join(srt_lines)



@app.post("/video-dubbing")
async def video_dubbing(
    video: UploadFile = File(...),
    sourceLang: str = Form(...),
    targetLang: str = Form(...),
):
    temp_video_path = None
    temp_audio_path = None
    output_audio_path = None
    srt_path = None

    try:
        # 1. Save the uploaded video file
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        with open(temp_video_path, "wb") as f:
            # Reading and writing chunks of the file to avoid memory overload
            while chunk := await video.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)

        # 2. Extract audio from the video using ffmpeg
        temp_audio_path = tempfile.mktemp(suffix=".mp3")
        ffmpeg.input(temp_video_path).output(temp_audio_path).run(quiet=True)

        # 3. Transcribe the audio with Whisper (with word timestamps enabled)
        result = model.transcribe(temp_audio_path, language=sourceLang, word_timestamps=True)
        segments = result.get("segments", [])

        # 4. Create dubbed audio with TTS for each segment
        combined = AudioSegment.silent(duration=0)

        for seg in segments:
            text = seg["text"]
            start_ms = seg["start"] * 1000
            end_ms = seg["end"] * 1000
            duration = end_ms - start_ms

            # Generate TTS for the segment
            tts = gTTS(text=text, lang=targetLang)
            tts_path = tempfile.mktemp(suffix=".mp3")
            tts.save(tts_path)
            audio_seg = AudioSegment.from_file(tts_path)

            # Adjust duration (pad or trim to match original segment duration)
            if len(audio_seg) < duration:
                audio_seg += AudioSegment.silent(duration=duration - len(audio_seg))
            else:
                audio_seg = audio_seg[:duration]

            # Add silence before this segment if needed
            if len(combined) < start_ms:
                combined += AudioSegment.silent(duration=start_ms - len(combined))

            combined += audio_seg

        # 5. Save the final dubbed audio file
        output_audio_path = tempfile.mktemp(suffix=".mp3")
        combined.export(output_audio_path, format="mp3")

        # 6. Generate the SRT subtitle file
        srt_content = segments_to_srt(segments)
        srt_path = tempfile.mktemp(suffix=".srt")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        # 7. Create a zip file containing the dubbed audio and subtitle file
        zip_path = tempfile.mktemp(suffix=".zip")
        with ZipFile(zip_path, "w") as zipf:
            zipf.write(output_audio_path, arcname="dubbed_audio.mp3")
            zipf.write(srt_path, arcname="subtitles.srt")

        # 8. Return the zip file containing dubbed audio and subtitles
        return FileResponse(zip_path, media_type="application/zip", filename="dubbed_video_with_subtitles.zip")

    except Exception as e:
        print("🔥 Unhandled exception:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during dubbing process: {str(e)}")

    finally:
        # Cleanup temporary files
        try:
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if output_audio_path and os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            if srt_path and os.path.exists(srt_path):
                os.remove(srt_path)
        except Exception as cleanup_error:
            print(f"Error cleaning up temporary files: {cleanup_error}")






# -----------------------------------------------------------------------------------------------------------------------------------------
@app.websocket("/dub-live")
async def live_dub(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

    try:
        while True:
            # Receive raw audio data from the client
            audio_data = await websocket.receive_bytes()
            print("🔊 Received audio chunk")

            # Process the received audio
            # Here you can transcribe, translate, and generate TTS based on the audio chunk.

            # For demonstration purposes, we will use Whisper to transcribe and gTTS for dubbing.
            temp_audio_path = tempfile.mktemp(suffix=".mp3")
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)

            # Transcribe the audio
            transcription_result = model.transcribe(temp_audio_path)
            transcription = transcription_result.get("text", "").strip()

            if transcription:
                print(f"📝 Transcription: {transcription}")

                # Generate dubbed audio using Google TTS
                tts = gTTS(text=transcription, lang='en')  # Change to your target language
                dubbed_audio_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(dubbed_audio_fp.name)

                # Send the dubbed audio back to the client
                with open(dubbed_audio_fp.name, "rb") as f:
                    dubbed_audio = f.read()
                    await websocket.send_bytes(dubbed_audio)
                os.remove(dubbed_audio_fp.name)

            os.remove(temp_audio_path)

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")

    except Exception as e:
        print(f"❌ Error during live dubbing: {e}")

    finally:
        await websocket.close()
        print("🔌 WebSocket closed")