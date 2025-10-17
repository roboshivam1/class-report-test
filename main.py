import streamlit as st
import whisper
import google.generativeai as genai
import tempfile


# ---------- CONFIG ----------
GEMINI_API_KEY = "AIzaSyBxHQro_i4Lh0V1Yf4bUIDkM6wwlxjB60Y"
genai.configure(api_key=GEMINI_API_KEY)
subjects = ["Physics", "Chemistry", "Maths", "Environmental Studies", "Science", "Social Studies", "English", "Hindi", "Artificial Intelligence", "Computer Science", "Informatics Practices", "Information Technology", "Biology", "History", "Political Science", "Economics", "Geography", "French", "Sanskrit", "Accountancy", "Business Studies", "Psychology", "Physical Education", "Robotics", "Yoga", "Art", "Music"]
sections = ["A", "B", "C", "D", "E", "Science", "Commerce", "Humanities"]
subjects.sort()

st.title("🎙️ Class Data Management System")

# ---------- FORM ----------
st.subheader("Enter Class Details")     

with st.form("credentials", enter_to_submit=False):
    teacher_name = st.text_input("Teacher Name")
    period_number = st.number_input("Period No.", min_value=1, step=1, max_value=8)
    subject = st.selectbox("Subject", subjects)
    start_time = st.time_input("Start Time")
    grade = st.number_input("Class", min_value=1, step=1, max_value=12)
    section = st.selectbox("Section", sections)
    submitted = st.form_submit_button("Submit Details")

# ---------- RECORD / UPLOAD AUDIO ----------
st.subheader("Record or Upload Class Audio")
st.info("🎤 Record a short class clip or upload an audio file (WAV, MP3, etc.)")

audio_input = st.audio_input("Record Class Audio")

# ---------- TRANSCRIPTION ----------
model = whisper.load_model("base")

def transcribe_with_whisper(audio_file):
    st.info("Transcribing with Whisper... ⏳")
    result = model.transcribe(audio_file, task='transcribe')
    return result["text"]

# ---------- GEMINI ANALYSIS ----------
def analyze_with_gemini(transcript):
    st.info("Analyzing with Gemini... 🧠")
    model_g = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
You are a classroom summarizer AI.

Here is a transcript of a class:

{transcript}

Please extract the following:
1. 📘 Topic taught
2. Subtopics taught
3. Exactly what all things were done by the teacher (e.g. theory, questions, test)
4. 📌 Summary of the class (2/3 paragraphs)
5. 📝 Homework assigned (if any)
6. ✅ Important bullet points
7. Relevance of the explanations to the topic
8. Any irregularities or problems in the teaching

Respond clearly and well-formatted.
"""
    response = model_g.generate_content(prompt)
    return response.text

# ---------- SAVE FUNCTION ----------
def save_output(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Teacher Name: {teacher_name}\n")
        f.write(f"Subject: {subject}\n")
        f.write(f"Period No.: {period_number}\n")
        f.write(f"Start Time: {start_time}\n")
        f.write(f"Class: {grade} \n")
        f.write(f"Section: {section} \n\n")
        f.write(text)

# ---------- MAIN ----------
if submitted:
    if audio_input is None:
        st.error("Please record or upload the class audio before submitting.")
    else:
        # Convert audio input (UploadedFile) to a temp file for Whisper
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_input.read())
            tmp.flush()
            audio_path = tmp.name

        transcript = transcribe_with_whisper(audio_path)
        st.subheader("🗒️ Transcript")
        st.write(transcript)

        save_output(transcript, "transcript_whisper.txt")

        report = analyze_with_gemini(transcript)
        st.subheader("📊 AI Class Report")
        st.write(report)

        save_output(report, "class_report_whisper.txt")

        st.success("✅ Report generated and saved successfully!")

        # Optional: Allow download
        st.download_button("⬇️ Download Report", data=report, file_name="class_report_whisper.txt")
