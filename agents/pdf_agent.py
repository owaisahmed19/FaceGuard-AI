import os
import io
import requests
import datetime
from fpdf import FPDF
from loguru import logger

class PdfAgent:
    """
    Handles fetching logs, passing them to generative AI for an executive summary,
    and rendering them down to a downloadable PDF.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        
    def _generate_executive_summary(self, events) -> str:
        if not self.api_key or not events:
            return "No generative AI summary available or no events recorded yet."
            
        # Compile a simplified list of events to not blow up token limits
        summary_lines = []
        for e in events[:50]:
            summary_lines.append(f"[{e.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {e.person_name} via {e.source}")
            
        raw_text = "\n".join(summary_lines)
        system_prompt = (
            "You are an elite automated security analyst for FaceGuard. "
            "Below is a list of recent face recognition events. "
            "Write a very professional, 1-paragraph Executive Summary describing the traffic and any unknown visitors. "
            "Do NOT use markdown. Assume the user is reading this on a printed PDF. Start with 'Executive Summary:'"
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Logs to summarize:\n{raw_text}"}
            ],
            "temperature": 0.5
        }
        
        try:
            resp = requests.post(self.api_url, headers=headers, json=data, timeout=12)
            if resp.status_code == 200:
                # Need to encode correctly or strip complex emojis for FPDF
                text = resp.json()["choices"][0]["message"]["content"].strip()
                return text.encode('latin-1', 'ignore').decode('latin-1')
            return f"API returned status {resp.status_code}"
        except Exception as e:
            logger.error(f"Failed to generate PDF summary: {e}")
            return "Error communicating with AI service."

    def create_pdf_report(self, events) -> bytes:
        # Generate the AI summary
        ai_summary = self._generate_executive_summary(events)
        
        class PDF(FPDF):
            def header(self):
                self.set_font("helvetica", "B", 15)
                self.cell(0, 10, "FaceGuard Intelligence - System Log Report", ln=True, align="C")
                self.ln(5)
                
            def footer(self):
                self.set_y(-15)
                self.set_font("helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")
                
        pdf = PDF()
        pdf.add_page()
        
        # 1. Title and date
        pdf.set_font("helvetica", "B", 10)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 10, f"Generated On: {current_time}", ln=True)
        pdf.ln(5)
        
        # 2. Executive Summary
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "AI Executive Summary", ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.multi_cell(0, 6, ai_summary)
        pdf.ln(10)
        
        # 3. Log Table
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Recent Recognition Events", ln=True)
        pdf.ln(2)
        
        pdf.set_font("helvetica", "B", 9)
        # Table Header
        pdf.cell(45, 8, "Timestamp", border=1)
        pdf.cell(45, 8, "Identity", border=1)
        pdf.cell(30, 8, "Source", border=1)
        pdf.cell(70, 8, "Report Snippet", border=1, ln=True)
        
        pdf.set_font("helvetica", "", 8)
        
        for e in events:
            time_str = e.timestamp.strftime('%Y-%m-%d %H:%M')
            # remove emojis/complex unicode for primitive FPDF helvetica
            name = str(e.person_name)[:18].encode('latin-1', 'ignore').decode('latin-1')
            source = str(e.source)[:12].encode('latin-1', 'ignore').decode('latin-1')
            conf = f"{e.confidence * 100:.1f}%" if str(e.person_name).lower() != "unknown" else "N/A"
            
            rep = str(e.report_text) if e.report_text else ""
            rep_snippet = (rep.replace('\n', ' ')[:45] + "...") if len(rep) > 45 else rep.replace('\n', ' ')
            rep_snippet = rep_snippet.encode('latin-1', 'ignore').decode('latin-1')
            
            pdf.cell(45, 8, time_str, border=1)
            pdf.cell(45, 8, name, border=1)
            pdf.cell(30, 8, source, border=1)
            pdf.cell(70, 8, rep_snippet, border=1, ln=True)
            
        # Return raw bytes
        return pdf.output()
