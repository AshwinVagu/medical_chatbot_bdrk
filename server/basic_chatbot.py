

import boto3
import json



def create_chatbot(prompt=""):
    client = boto3.client('bedrock-runtime')

    
    
    # Prepare the body with the prompt
    body = json.dumps({
        'inputText':prompt,
        'textGenerationConfig':{
             'maxTokenCount':300,
             'stopSequences':[],
             'temperature':1,
             'topP':1
             }
    })
    
    # Invoke the model
    response = client.invoke_model(
        modelId='model-id',  # Replace with your specific model ID
        body=body,  # Pass the JSON string
        contentType='application/json',
        accept='application/json'
    )
    
    # Read the StreamingBody and parse it as JSON
    response_body = response['body'].read().decode('utf-8')
    response_json = json.loads(response_body)

    
    # Extract the generated text
    generated_text = response_json['results'][0]['outputText']

    
    return generated_text

if __name__ == "__main__":
    initial_call = True
    while True:
        if initial_call==False:  
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break 
            # Based on the type of LLM model used the prompt must be given in such a way.
            response = create_chatbot(user_input)

            print(f"Chatbot: {response}")

        else:
            # Based on the type of LLM model used the prompt must be given in such a way.
            system_prompt = "You are a highly knowledgeable and empathetic AI chatbot designed primarily for medical purposes. Your primary role is to provide accurate, up-to-date, and reliable information on various medical topics, including but not limited to symptoms, conditions, treatments, medications, and general health advice. Your responses should be based on evidence-based medical guidelines and current medical knowledge. While your primary focus is on medical-related conversations, you are also capable of engaging in casual and friendly conversations when appropriate. Your tone should be professional yet approachable, ensuring that users feel comfortable and supported in their interactions with you. In all interactions, prioritize user safety, confidentiality, and clarity. Also, Use the AWS Knowledge Base to get doctor timings and the tests they perform for a hospital. Keep your responses as short and discrete as possible. Only give answers from your side and don't answer for the user."
            response = create_chatbot(system_prompt)   
            initial_call = False  








