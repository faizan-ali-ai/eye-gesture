"""
OmniControl AI - Proctoring Engine Backend
All-in-one single-file Python FastAPI server modified for Vercel Serverless.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import cv2
import os
import time
import json
import asyncio
from typing import Optional, List
from fastapi.responses import HTMLResponse

app = FastAPI(title="OmniControl AI Proctoring Backend")

# 1. CORS Middleware (Essential to fix browser blocking issues!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows local index.html to communicate freely
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FRONTEND ROUTE ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # File ka absolute path nikalna taake Vercel par error na aaye
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(BASE_DIR, "single_index.html")
    
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Frontend File Not Found</h1><p>{str(e)}</p>", status_code=404)

# Global In-Memory Stores
logs_db = []
detailed_logs_db = []

class LogEntry(BaseModel):
    startTime: float
    duration: float
    status: str

class GenerateReportRequest(BaseModel):
    candidateName: str
    interviewId: Optional[str] = "General Profile"
    date: str
    durationSeconds: float
    focusedPercentage: float
    totalViolations: int
    events: List[LogEntry]

# --- Home & Health Stats ---
@app.get("/")
async def root():
    return {
        "status": "Online",
        "engine": "OmniControl AI Proctoring Node",
        "timestamp": time.time(),
        "active_logs_count": len(logs_db)
    }

# --- System Audit Logs ---
@app.get("/logs")
async def get_logs():
    return {
        "total_distractions": len(logs_db),
        "logs": logs_db,
        "detailed": detailed_logs_db
    }

@app.post("/api/sync-log-entry")
async def sync_log_entry(entry: LogEntry):
    local_struct = time.localtime(entry.startTime / 1000.0)
    timestamp = time.strftime("%H:%M:%S", local_struct)
    
    action_label = entry.status.replace("Looking ", "").replace(" (Away)", "")
    log_string = f"[{timestamp}] - {action_label} for {entry.duration:.1f}s"
    
    logs_db.insert(0, log_string)
    detailed_logs_db.insert(0, {
        "startTime": entry.startTime,
        "duration": entry.duration,
        "status": entry.status,
        "text": log_string
    })
    
    return {"success": True, "entry": log_string}

@app.post("/api/clear-logs")
async def clear_logs():
    global logs_db, detailed_logs_db
    logs_db.clear()
    detailed_logs_db.clear()
    return {"success": True, "message": "In-memory telemetry logs cleared."}

# --- Gemini AI Report Generation ---
@app.post("/api/generate-summary")
async def generate_summary(data: GenerateReportRequest):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "GEMINI_API_KEY environment variable is not defined on the server host."}
            )
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        logs_markdown = ""
        if data.events:
            for idx, e in enumerate(data.events):
                offset = int((e.startTime - data.events[-1].startTime) / 1000.0)
                logs_markdown += f"- [Offset {offset:+}s] Incident: {e.status}, Duration: {e.duration:.1f} seconds\n"
        else:
            logs_markdown = "No gaze incidents or look-away violations were recorded. Candidate maintained full compliance."

        prompt = f"""
You are an AI Proctoring Evaluation Agent. Review the following proctoring session gaze data and generate a professional compliance report.

### Candidate & Session Metadata:
- Candidate Name: {data.candidateName}
- Target Job / Interview Profile: {data.interviewId}
- Session Timestamp: {data.date}
- Monitoring Session Duration: {data.durationSeconds:.1f} seconds
- Eye Gaze Compliance Index: {data.focusedPercentage:.1f}% 
- Total Violation Incidents Flagged: {data.totalViolations}

### Gaze Incidents Audit Train:
{logs_markdown}

Please generate an evaluation report with these exact sections in clean Markdown format:
1. **Gaze Compliance Evaluation**: A concise analysis of the candidate's visual Focus Index ({data.focusedPercentage:.1f}%) with appropriate professional tone.
2. **Deviation Patterns & Absence Analysis**: Analyze the logged events showing lookaways or head attitude shifts (e.g., specific gaze-offs, look-left/right, head turns, head tilts, blinking, or face absent entirely) and interpret what these behaviors typically suggest (e.g., possible prompts reading, side monitor distraction, or simply contemplating).
3. **Proctoring Verdict**: Provide a clear proctoring verdict of either:
   - **SATISFACTORY FOCUS** (Index >= 80%, minor natural gaze shifts)
   - **MARGINAL FOCUS ALERT** (Index between 60%-79% or repeating gaze deviations)
   - **HIGH COMPLIANCE CONCERN** (Index < 60% or long persistent absences / off-screen looking)
4. **Actionable Hiring Recommendation**: Give constructive, humble hiring panel guidance for the interview debrief.

Ensure your tone is completely objective and professional. Do not write generic greetings/salutations in the output.
"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        return {"summary": response.text}

    except ImportError:
        return JSONResponse(
            status_code=500,
            content={"error": "Please install the generative AI model libraries: 'pip install google-generativeai' to run AI summaries."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred on the Python host while generating your report: {str(e)}"}
        )

# --- OpenCV Video Feed Route (Vercel Safe Option) ---
camera = None

def get_camera_frame():
    global camera
    
    # VERIFICATION: Agar code Vercel cloud par chal raha hai to actual camera open na karein (taake server crash na ho)
    if os.environ.get("VERCEL") == "1":
        dummy_frame = cv2.Mat.zeros(480, 640, cv2.CV_8UC3)
        cv2.putText(dummy_frame, "Live Stream Hosted (Cloud Node Active)", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', dummy_frame)
        return jpeg.tobytes()

    # Local environment ke liye camera handle
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    success, frame = camera.read()
    if not success:
        dummy_frame = cv2.Mat.zeros(480, 640, cv2.CV_8UC3)
        cv2.putText(dummy_frame, "Webcam capture failed or device busy", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', dummy_frame)
        return jpeg.tobytes()
        
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def gen_frames():
    while True:
        frame_bytes = get_camera_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_main:app", host="127.0.0.1", port=8000, reload=True)