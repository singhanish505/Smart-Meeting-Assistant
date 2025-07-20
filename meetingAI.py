##Meeting Assitant V1.0
# This is a simple Streamlit app that uses OpenAI's API to transcribe audio files
# and summarize meeting transcripts, extract action items, and decisions.
# It simulates the agentic roles of Perception, Reasoning, and Action in a  meeting context.
#using openai's API

import streamlit as st
import json
import os
import datetime
import uuid
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = st.secrets["OPENAI_API_KEY"]

#api_key2 = os.getenv("OPEN_API_KEY")
client = OpenAI(api_key=api_key)
#st.secrets["openai"]["api_key"]
#openai_key = st.secrets["openai"]["api_key"]
#openai_key = st.secrets.get("openai", {}).get("api_key")


if not client:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
    st.stop()

api_key = client

# Directory to store processed meeting data
OUTPUT_DIR = "meeting_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("temp_audio", exist_ok=True) # Ensure temp_audio directory exists

# --- Helper Functions for Agentic Modules ---

def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using OpenAI Whisper API.
    Agentic Role: Perception (converting audio to text).
    """
    logging.info(f"Starting transcription for: {audio_file_path}")
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text" # Request plain text for simplicity
            )
        logging.info("Transcription successful.")
        return transcript
    except Exception as e:
        logging.error(f"OpenAI API Error during transcription: {e}")
        st.error(f"Error during transcription: {e.message}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during transcription: {e}")
        st.error(f"An unexpected error occurred during transcription: {e}")
        return None

def summarize_text(text, meeting_title="a meeting"):
    """
    Summarizes the given text using an OpenAI LLM, with optional meeting context.
    Agentic Role: Reasoning (distilling information).
    """
    logging.info("Starting summarization.") 
    system_prompt = f"You are a helpful meeting assistant. Summarize the following transcript from a meeting titled '{meeting_title}' concisely, highlighting key discussion points, decisions made, and overall outcomes. Focus on the most important information."
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=500
        )
        summary = response.choices[0].message.content
        logging.info("Summarization successful.")
        return summary
    except Exception as e:
        logging.error(f"OpenAI API Error during summarization: {e}")
        st.error(f"Error during summarization: {e.message}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during summarization: {e}")
        st.error(f"An unexpected error occurred during summarization: {e}")
        return None

def extract_action_items_and_decisions(text, meeting_title="a meeting"):
    """
    Extracts action items and decisions from the text using an OpenAI LLM, with optional meeting context.
    Agentic Role: Action (identifying actionable insights).
    """
    logging.info("Starting action item and decision extraction.")
    system_prompt = f"""
    You are an intelligent assistant designed to extract action items and key decisions from the transcript of a meeting titled '{meeting_title}'.
    For each action item, identify the task, the person responsible (if mentioned), and any deadline (if mentioned).
    For each decision, briefly state the decision.
    If no action items or decisions are found, return empty lists.

    Provide the output in a JSON format with two keys: "action_items" (a list of objects) and "decisions" (a list of strings).
    Example for action_items: [{{"task": "Follow up with client", "assignee": "John", "deadline": "EOD Friday"}}]
    Example for decisions: ["Approved the new marketing budget", "Decided to postpone the launch"]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # You can try "gpt-4" for better quality if available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        extraction_output = json.loads(response.choices[0].message.content)
        logging.info("Extraction successful.")
        return extraction_output.get("action_items", []), extraction_output.get("decisions", [])
    except client as e:
        logging.error(f"OpenAI API Error during extraction: {e}")
        st.error(f"Error during extraction: {e.message}")
        return [], []
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from LLM response: {e}. Raw response: {response.choices[0].message.content}")
        st.warning("Could not parse action items/decisions. The AI might not have returned valid JSON.")
        return [], []
    except Exception as e:
        logging.error(f"An unexpected error occurred during extraction: {e}")
        st.error(f"An unexpected error occurred during extraction: {e}")
        return [], []

def save_meeting_data(meeting_id, meeting_title, transcript, summary, action_items, decisions):
    """
    Saves the processed meeting data to a local JSON file.
    Agentic Role: Memory (persisting information).
    """
    file_path = os.path.join(OUTPUT_DIR, f"{meeting_id}.json")
    data = {
        "meeting_id": meeting_id,
        "meeting_title": meeting_title,
        "timestamp": datetime.datetime.now().isoformat(),
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions
    }
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"Meeting data saved to {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving meeting data: {e}")
        st.error(f"Error saving meeting data: {e}")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="Smart Meeting Assistant (MVP)", layout="wide")

st.title("🤖 Smart Meeting Assistant (MVP)")
st.markdown("Provide meeting details and upload an audio file to get a summary, action items, and key decisions.")

# Input for meeting context (simulating detection/joining)
meeting_title = st.text_input("Meeting Title (e.g., 'Weekly Sync', 'Project Alpha Review')", "Please provide a title for the meeting held")

uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    # Generate a unique ID for this meeting session
    meeting_id = str(uuid.uuid4())
    temp_audio_path = os.path.join("temp_audio", f"{meeting_id}_{uploaded_file.name}")

    with st.spinner("Processing audio... This might take a moment."):
        # Save the uploaded file temporarily
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info(f"Temporary audio file saved: {temp_audio_path}")

        # 1. Transcribe Audio
        st.subheader("1. Transcribing Audio...")
        transcript = transcribe_audio(temp_audio_path)

        if transcript:
            st.success("Transcription Complete!")
            with st.expander("View Full Transcript"):
                st.write(transcript)

            # 2. Summarize Text (using meeting title for context)
            st.subheader("2. Generating Summary...")
            summary = summarize_text(transcript, meeting_title)

            if summary:
                st.success("Summary Generated!")
                st.info(summary)

                # 3. Extract Action Items and Decisions (using meeting title for context)
                st.subheader("3. Extracting Action Items & Decisions...")
                action_items, decisions = extract_action_items_and_decisions(transcript, meeting_title)

                if action_items or decisions:
                    st.success("Extraction Complete!")
                    if action_items:
                        st.markdown("#### Action Items:")
                        for item in action_items:
                            # Ensure 'task' key exists before accessing
                            task_str = item.get('task', 'N/A')
                            assignee_str = f" (Assigned to: {item['assignee']})" if item.get('assignee') else ""
                            deadline_str = f" (Deadline: {item['deadline']})" if item.get('deadline') else ""
                            st.write(f"- **{task_str}**{assignee_str}{deadline_str}")
                    if decisions:
                        st.markdown("#### Key Decisions:")
                        for decision in decisions:
                            st.write(f"- {decision}")
                else:
                    st.warning("No specific action items or decisions were extracted.")

                # Save all processed data
                st.subheader("Saving Meeting Data...")
                saved_file = save_meeting_data(meeting_id, meeting_title, transcript, summary, action_items, decisions)
                if saved_file:
                    st.success(f"Meeting data saved successfully to: `{saved_file}`")
                else:
                    st.error("Failed to save meeting data.")

        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logging.info(f"Cleaned up temporary audio file: {temp_audio_path}")


#st.markdown("---")
#st.markdown("""
#**Note on Automatic Meeting Detection & Joining:**

#For a hackathon MVP, truly automatic detection and joining of live meetings from platforms like Teams, Zoom, or Slack, along with real-time audio capture, is a significant technical challenge. It would typically involve:
#* **Platform-Specific APIs/SDKs:** Each platform has its own set of APIs for scheduling, joining, and sometimes capturing audio, often requiring complex authentication (OAuth) and permissions.
#* **Virtual Audio Devices:** To capture the audio stream directly from the meeting application on your system.
#* **Real-time Streaming:** Handling continuous audio input and processing it in chunks.
#This MVP focuses on the core agentic processing (Perception via transcription, Reasoning via summarization, Action via extraction) once the audio is provided. Future enhancements could explore deeper integrations and real-time capabilities.
#""")
#'''
