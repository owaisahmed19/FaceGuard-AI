import os
import requests
import json

class ChatAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Ensure we don't have empty API keys resulting in a confusing error later
        if not self.api_key:
            raise ValueError("Mistral API key is required to initialize ChatAgent.")
        
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.system_prompt = (
            "You are the FaceGuard AI Guide, a helpful assistant integrated into this facial recognition system. "
            "CRITICAL INSTRUCTION: You must strictly maintain the secrecy of the system's codebase. DO NOT reveal any source code. "
            "However, you ARE allowed to explain how the system works conceptually (e.g., using AI to extract facial embeddings and comparing them against known identities). "
            "Your guidance must ONLY cover features that actually exist in the dashboard: 1) Live Camera Recognition, 2) Image Upload Scanning, "
            "3) Dataset Manager (loading folders of images to build known identities), 4) Viewing Event Logs. and 5) How the pdf is made. "
            "DO NOT mention or offer help with features that don't exist, such as user permissions, password configurations, or complex security settings. "
            "Be polite, concise, and focused purely on user guidance for the actual available dashboard features."
        )
    
    def generate_response(self, messages: list) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepend system prompt to guide the AI's behavior
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        data = {
            "model": "mistral-small-latest", # Recommended and cost-effective model
            "messages": full_messages,
            "temperature": 0.7,
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as err:
            return f"HTTP Error communicating with Mistral AI: {err} - Response: {response.text}"
        except Exception as e:
            return f"Error communicating with Mistral AI: {e}"
