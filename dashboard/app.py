import sys
import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st

sys.path.insert(0, ".")

from utils.database import init_db, log_recognition_event, get_recent_events
from agents.dataset_agent import DatasetAgent
from agents.recognition_agent import RecognitionAgent
from agents.report_agent import ReportAgent
from agents.chat_agent import ChatAgent

# Initialize database
init_db()

# --- Page Config
st.set_page_config(
    page_title="FaceGuard Intelligence",
    page_icon="🛡",
    layout="wide"
)

# --- Session State
if "dataset_agent" not in st.session_state:
    da = DatasetAgent()
    da.load_embeddings()
    st.session_state.dataset_agent = da

if "recognition_agent" not in st.session_state:
    st.session_state.recognition_agent = RecognitionAgent(st.session_state.dataset_agent)

if "report_agent" not in st.session_state:
    # Share the same Mistral API key for generative logging
    st.session_state.report_agent = ReportAgent(api_key="M3TVWJ8fiiVkvZiXqgFj7850PiuD0tak")

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = ChatAgent(api_key="M3TVWJ8fiiVkvZiXqgFj7850PiuD0tak")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Welcome! I am the FaceGuard AI Guide. I am here to help you navigate and use the system. How can I assist you today?"}
    ]

# --- Helper Methods
def draw_results(image_bgr, results):
    """Draws bounding boxes and labels on the image."""
    for face in results:
        top, right, bottom, left = face['box']
        name = face['name']
        conf = face['confidence'] * 100
        
        # Color: Green if known, Red if unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw box
        cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
        
        # Draw label (just the name)
        label = f"{name}" if name != "Unknown" else "Unknown"
        cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(image_bgr, label, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    return image_bgr

def process_and_log(image_rgb, image_bgr, source, display_placeholder=None, report_placeholder=None):
    """Core pipeline: recognize, report, log, display."""
    # 1. Recognize
    results = st.session_state.recognition_agent.process_frame(image_rgb)
    
    # 2. Draw
    drawn_bgr = draw_results(image_bgr, results)
    
    # 3. Report
    report = st.session_state.report_agent.generate_report(results, source)
    
    # 4. Log
    if results: # Only log if faces were detected
        for face in results:
            log_recognition_event(
                source=source,
                person_name=face['name'],
                confidence=face['confidence'],
                report_text=report['text'],
                report_json=report['json']
            )
            
    # 5. Display
    if display_placeholder:
        # Convert BGR back to RGB for Streamlit
        drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
        display_placeholder.image(drawn_rgb, channels="RGB", use_container_width=True)
        
    if report_placeholder and results:
        report_placeholder.code(report['text'], language="text")

# --- UI Setup
st.title("🛡️ FaceGuard Intelligence")
st.markdown("Real-time facial recognition and smart reporting system.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎥 Live Camera", "🖼️ Upload Image", "⚙️ Dataset Manager", "📜 Logs", "💬 System Chat"])

# ========================
# TAB 1: Live Camera
# ========================
with tab1:
    st.header("Live Camera Recognition")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Controls")
        if st.button("▶️ Start Camera", type="primary", use_container_width=True):
            st.session_state.camera_active = True
        
        if st.button("⏹️ Stop Camera", use_container_width=True):
            st.session_state.camera_active = False

    with col1:
        frame_placeholder = st.empty()
        live_report_placeholder = st.empty()
        
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not open webcam.")
                st.session_state.camera_active = False
            else:
                st.success("Camera is active. Press Stop to halt.")
                
                frame_count = 0
                last_results = []
                last_report = None
                last_seen_identities = set()
                
                # We use a loop while boolean is True. 
                # Streamlit naturally restricts `while` loops without rerun tricks,
                # but an inline while loop can work if we handle it carefully.
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam.")
                        break
                    
                    # Convert BGR to RGB for processing
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Optimize: Process AI only every 5th frame for CPU efficiency
                    if frame_count % 5 == 0:
                        last_results = st.session_state.recognition_agent.process_frame(rgb_frame)
                        
                        if last_results:
                            current_identities = set([f['name'] for f in last_results])
                            
                            # ONLY trigger the heavy Generative AI report and DB log if identities change
                            if current_identities != last_seen_identities:
                                last_report = st.session_state.report_agent.generate_report(last_results, "Live Camera")
                                for face in last_results:
                                    log_recognition_event(
                                        source="Live Camera",
                                        person_name=face['name'],
                                        confidence=face['confidence'],
                                        report_text=last_report['text'],
                                        report_json=last_report['json']
                                    )
                                last_seen_identities = current_identities
                    
                    # Always draw the last known results directly to keep video smooth
                    drawn_bgr = draw_results(frame, last_results)
                    drawn_rgb = cv2.cvtColor(drawn_bgr, cv2.COLOR_BGR2RGB)
                    
                    frame_placeholder.image(drawn_rgb, channels="RGB", use_container_width=True)
                    
                    if last_report and last_results:
                        live_report_placeholder.code(last_report['text'], language="text")
                    
                    frame_count += 1
                    
                    # Brief sleep to keep UI responsive
                    time.sleep(0.02)
                
                cap.release()

# ========================
# TAB 2: Upload Image
# ========================
with tab2:
    st.header("Upload Image for Recognition")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Read the file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        col_img, col_rep = st.columns([2, 1])
        with col_img:
            img_ph = st.empty()
        with col_rep:
            st.markdown("### Agent Report")
            rep_ph = st.empty()
            
        process_and_log(
            image_rgb=img_rgb,
            image_bgr=img_bgr.copy(),
            source="Image Upload",
            display_placeholder=img_ph,
            report_placeholder=rep_ph
        )

# ========================
# TAB 3: Dataset Manager
# ========================
with tab3:
    st.header("Dataset Embeddings Manager")
    st.markdown("""
    When you add new folders and images to `data/people_dataset/`, you need to rebuild the embeddings.
    """)
    
    st.write(f"Current Known Identities: **{len(set(st.session_state.dataset_agent.known_face_names))}**")
    
    if st.button("🔄 Rebuild Dataset Embeddings", type="primary"):
        with st.spinner("Scanning dataset and building encodings... this may take a moment."):
            success = st.session_state.dataset_agent.rebuild_embeddings()
            if success:
                st.success("Successfully rebuilt embeddings!")
                # Reload recognition agent with new dataset
                st.session_state.recognition_agent = RecognitionAgent(st.session_state.dataset_agent)
            else:
                st.error("Failed to build embeddings. Make sure the dataset directory possesses folders with images.")

# ========================
# TAB 4: Logs
# ========================
with tab4:
    import datetime
    from agents.pdf_agent import PdfAgent
    st.header("Recognition Event Logs")
    
    col1, col2 = st.columns([1,2])
    with col1:
        if st.button("🔄 Refresh Logs"):
            pass # Streamlit reruns
            
    events = get_recent_events(limit=50)
    
    with col2:
        if events:
            if st.button("📝 Generate Gen-AI PDF Report", use_container_width=True):
                with st.spinner("Mistral AI is analyzing logs and building PDF..."):
                    pdf_agent = PdfAgent(api_key="M3TVWJ8fiiVkvZiXqgFj7850PiuD0tak")
                    st.session_state.pdf_bytes = pdf_agent.create_pdf_report(events)
                    st.session_state.pdf_name = f"FaceGuard_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
            if "pdf_bytes" in st.session_state:
                st.download_button(
                    label="📥 Download Output PDF",
                    data=bytes(st.session_state.pdf_bytes),
                    file_name=st.session_state.pdf_name,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
        
    if not events:
        st.info("No events logged yet.")
    else:
        for event in events:
            with st.expander(f"[{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {event.person_name} ({event.source})"):
                col_info, col_json = st.columns(2)
                with col_info:
                    st.markdown(f"**Identity:** {event.person_name}")
                    st.markdown(f"**Source:** {event.source}")
                    st.markdown("**Report:**")
                    st.text(event.report_text)
                    
                with col_json:
                    st.markdown("**Structured JSON:**")
                    st.json(event.report_json)

# ========================
# TAB 5: System Chat
# ========================
with tab5:

    # Container for chat messages that dynamically grows with content
    # Streamlit's default chat styling closely mimics ChatGPT & Gemini natively!
    chat_container = st.container(key="chat_container")
    
    with chat_container:
        # Display chat messages from history on app rerun
        for message in st.session_state.chat_messages:
            avatar = "✨" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                
    # React to user input
    if prompt := st.chat_input("Message FaceGuard AI..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            # Render user message live
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            with st.spinner("AI is thinking..."):
                response = st.session_state.chat_agent.generate_response(st.session_state.chat_messages)
                
            # Render assistant message live
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

