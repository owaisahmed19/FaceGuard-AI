import json
import datetime
import requests
from loguru import logger

class ReportAgent:
    """
    Generates structured English reports and JSON data for face recognition events.
    Uses Mistral AI to generate fluid text summaries if an API key is provided.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def generate_report(self, recognition_results, source_type: str):
        """
        Creates an English report based on the results from the RecognitionAgent.
        
        Args:
            recognition_results (list of dict): Detected faces and their matches.
            source_type (str): "Live Camera" or "Uploaded Image". 
            
        Returns:
            dict: Contains 'text' (human readable) and 'json' (structured) outputs.
        """
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        num_faces = len(recognition_results)
        
        # 1. Structure JSON
        report_data = {
            "timestamp": timestamp_str,
            "source": source_type,
            "faces_detected": num_faces,
            "matches": []
        }
        
        for face in recognition_results:
            report_data["matches"].append({
                "identity": face['name'],
                "confidence": round(face['confidence'], 4),
                "box": face['box']
            })
            
        json_output = json.dumps(report_data, indent=2)
        
        # 2. Build human readable text
        text_lines = [
            f"=== 🛡 FaceGuard Recognition Report ===",
            f"Date / Time    : {timestamp_str}",
            f"Source Type    : {source_type}",
            f"Faces Detected : {num_faces}",
            f"---------------------------------------"
        ]
        
        if num_faces == 0:
            text_lines.append("Analysis Complete. No faces were detected in the source.")
        else:
            for idx, face in enumerate(recognition_results, start=1):
                name = face['name']
                conf = face['confidence'] * 100
                
                if name == "Unknown":
                    text_lines.append(f"Face {idx}: Could not match identity. (Status: Unknown)")
                else:
                    text_lines.append(f"Face {idx}: Successfully identified as '{name}'.")
                    
            text_lines.append("\nSummary:")
            known = [f['name'] for f in recognition_results if f['name'] != "Unknown"]
            if known:
                text_lines.append(f"The system recognized: {', '.join(known)}.")
            else:
                text_lines.append("The system analyzed the frame but found no matches in the dataset.")
                
        text_output = "\n".join(text_lines)
        
        # 3. IF Generative AI is enabled, enhance the text_output using Mistral
        if self.api_key:
            try:
                # Strip confidence arrays before sending to Generative AI to completely block it
                mistral_data = {
                    "timestamp": timestamp_str,
                    "source": source_type,
                    "faces_detected": num_faces,
                    "identities_seen": [f['name'] for f in recognition_results]
                }
                
                system_prompt = (
                    "You are an elite, automated security analyst for FaceGuard. "
                    "Your job is to read the raw JSON of recognized individuals and write a concise, professional 1-3 sentence security alert summarizing who was seen. "
                    "Do NOT use markdown. Do NOT mention any metrics, statistics, percentages, or confidence scores. Start directly with 'Alert:' or 'Log:'. "
                )
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                data = {
                    "model": "mistral-small-latest",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Please summarize this event data: {json.dumps(mistral_data)}"}
                    ],
                    "temperature": 0.5
                }
                
                resp = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data, timeout=8)
                if resp.status_code == 200:
                    ai_text = resp.json()["choices"][0]["message"]["content"].strip()
                    # Prepend our fixed header to the AI text for standard log styling
                    text_output = (
                        f"=== 🛡 FaceGuard AI Recognition Report ===\n"
                        f"Date / Time    : {timestamp_str}\n"
                        f"Source Type    : {source_type}\n"
                        f"Faces Detected : {num_faces}\n"
                        f"---------------------------------------\n"
                        f"{ai_text}"
                    )
            except Exception as e:
                logger.error(f"Generative AI Reporting failed (falling back to standard format): {e}")
        
        return {
            "text": text_output,
            "json": json_output,
            "parsed": report_data
        }
