import google.generativeai as gen_ai

class GeminiAPI:
    def __init__(self, model):
        self.model = model

    def get_recommendation(self, user_prompt):
        """Send user's message to Gemini-Pro and get the response."""
        # Processing user input
        try:
            response = self.model.send_message(user_prompt)
            # Successfully received response from Gemini-Pro
            return response
        except Exception as e:
            # Failed to receive response from Gemini-Pro
            return type('Response', (object,), {"text": "Default recipe response"})()

if __name__ == "__main__":
    # Assuming the API key is configured elsewhere
    try:
        model = gen_ai.GenerativeModel('gemini-1.0-pro')
        # Connected to Gemini-Pro successfully
    except Exception as e:
        # Failed to connect to Gemini-Pro
        model = None

    # Define user_prompt here or retrieve it from user input
    user_prompt = ""

    gemini_api = GeminiAPI(model)
    response = gemini_api.get_recommendation(user_prompt)
