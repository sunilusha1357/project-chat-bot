import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

def get_gemini_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not configured. Add your real Gemini API key to the .env file."
        )
    return api_key

class GeminiRAGChat:
    def __init__(self):
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        
        # Startup check
        try:
            # Just listing models to verify connection
            list(genai.list_models())
            print("Gemini API connected OK")
        except Exception as e:
            print(f"Gemini API connection failed: {e}")
            raise e
        
        self.system_instruction = (
            "You MUST answer ONLY from the provided context. "
            "If the context does not contain the answer, respond with exactly: "
            "'I could not find that information in the uploaded documents.' "
            "Do not use your training data."
        )
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=self.system_instruction
        )
        self.chat = self.model.start_chat(history=[])

    def ask(self, user_question: str, context: str) -> str:
        prompt = (
            f"CONTEXT (answer only from this):\n{context}\n\n"
            f"User question: {user_question}\n\n"
            "IMPORTANT: Base your answer only on the context above."
        )
        response = self.chat.send_message(prompt)
        return response.text

    def reset(self):
        self.chat = self.model.start_chat(history=[])

    def get_history(self) -> list:
        history = []
        for message in self.chat.history:
            history.append({
                "role": message.role,
                "text": message.parts[0].text
            })
        return history
