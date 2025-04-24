from flask import Flask, request, jsonify
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import os

app = Flask(__name__)

# Initialize the client
endpoint = "https://models.github.ai/inference"
model = "deepseek/DeepSeek-V3-0324"
token = os.getenv("GITHUB_TOKEN")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# Psychology-specific system prompt
PSYCHOLOGY_SYSTEM_PROMPT = """
You are Dr. Mind, a compassionate and highly skilled AI psychologist with 30 years of experience. 
Your role is to provide supportive, ethical, and professional psychological guidance while maintaining professional boundaries.

Guidelines:
1. Always respond with empathy and understanding
2. Ask clarifying questions when needed
3. Never provide medical diagnoses
4. Recommend seeking professional help when appropriate
5. Maintain confidentiality (though remind users this is an AI service)
6. Use active listening techniques
7. Provide evidence-based psychological insights
8. Avoid giving direct advice, instead offer perspectives
"""

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    try:
        response = client.complete(
            messages=[
                SystemMessage(PSYCHOLOGY_SYSTEM_PROMPT),
                UserMessage(user_message),
            ],
            temperature=0.7,  # Slightly lower for more consistent responses
            top_p=0.9,       # Higher for more diversity in responses
            max_tokens=1024,  # Enough for detailed responses
            model=model
        )
        
        return jsonify({
            'response': response.choices[0].message.content
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
