from flask import Flask, request, render_template, jsonify
from utils.custom_preprocessing import chat
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
from dotenv import load_dotenv
import openai #mport OpenAI
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

load_dotenv("env")

# Initialize the OpenAI client
open_api_key = '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'
openai.api_key = open_api_key

# Initialize the API credentials
vendor_id = os.getenv("VendorId")
vendor_password = os.getenv("VendorPassword")
account_id = os.getenv("AccountId")
account_password = os.getenv("AccountPassword")

# Load the base model and tokenizer
base_model_name = "NousResearch/Llama-2-7b-chat-hf" 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model_ = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='auto', torch_dtype=torch.float16)

# Print the model structure for debugging
def print_model_structure(model):
    for name, module in model.named_modules():
        print(name)

print("Base model structure:")
print_model_structure(base_model_)

# Load the fine-tuned adapter weights
adapter_path = "./static/checkpoint-1070"
try:
    model = PeftModel.from_pretrained(base_model_, adapter_path)
except KeyError as e:
    print(f"KeyError encountered: {e}")
    print("Model structure at the time of error:")
    print_model_structure(base_model_)

app = Flask(__name__)

def identify_intent(user_query):
    prompt = (
       f"""Identify the intent of the following query: "{user_query}".
       Is it related to rescheduling an appointment, canceling an appointment, Book appointment or something else?"""
    )

    chat_completion = openai.ChatCompletion.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )

    # Extract the intent from the response
    intent = chat_completion.choices[0].message.content.strip().lower()
    print(intent)
    return intent

# Function to extract required information
def fetch_info(response):
    model = ChatOpenAI(temperature=0, openai_api_key=open_api_key)

    class Req_info(BaseModel):
        Lastname: str = Field(
            description="The value that identifies lastname of the user")
        DateOfBirth: str = Field(
            description="The value that represents date of birth")

    parser = PydanticOutputParser(pydantic_object=Req_info)
    prompt = PromptTemplate(
        template="Extract the information that defined.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser
    final_results = chain.invoke({"query": response})
    final_results1=dict(final_results)
    print(type(final_results1))
    return final_results1
    # return final_results.json()

# Function to get authentication token
def get_auth_token(vendor_id, vendor_password, account_id, account_password):
    auth_url = "https://iochatbot.maximeyes.com/api/v2/account/authenticate"
    auth_payload = {
        "VendorId": vendor_id,
        "VendorPassword": vendor_password,
        "AccountId": account_id,
        "AccountPassword": account_password
    }
    headers = {'Content-Type': 'application/json'}
    try:
        auth_response = requests.post(auth_url, json=auth_payload, headers=headers)
        auth_response.raise_for_status()
        response_json = auth_response.json()

        if response_json.get('IsToken'):
            return response_json.get('Token')
        else:
            print("Error message:", response_json.get('ErrorMessage'))
            return None
    except requests.RequestException as e:
        print(f"Authentication failed: {str(e)}")
        return None
    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        return None


# Function to reschedule appointment
def reschedule_appointment(auth_token, Lastname, DateOfBirth, IsActive="true"):

    headers = {
        'Content-Type': 'application/json',
        'apiKey': f'bearer {auth_token}'  # Corrected this line
    }

    # Step 1: Get the patient number
    get_patient_url = "https://iochatbot.maximeyes.com/api/patient/getlistofpatient"
    patient_payload = {
        "Lastname": Lastname,
        "DateOfBirth": DateOfBirth,
        "IsActive": IsActive
    }

    try:
        patient_response = requests.post(get_patient_url, json=patient_payload, headers=headers)
        patient_response.raise_for_status()
        print(f"URL: {get_patient_url}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return

    if patient_response.status_code != 200:
        return f"Failed to get patient details. Status code: {patient_response.status_code}"
    else:
        pass

    patient_number=input("enter the patient_number: ")
    # Step 1: Get the appointment details
    get_appointment_url = f"https://iochatbot.maximeyes.com/api/appointment/getappointmentchatbot?PatientNumber={patient_number}"

    try:
        appointment_response = requests.get(get_appointment_url, headers=headers)
        appointment_response.raise_for_status()

        print(f"URL: {get_appointment_url}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    if appointment_response.status_code != 200:
        return f"Failed to get appointment details. Status code: {appointment_response.status_code}"

    try:
        appointment_details = appointment_response.json()
    except ValueError:
        return "Failed to parse appointment details response as JSON."

    if not appointment_details:
        return "No appointments found for the given patient number."

    # Extract required IDs from the appointment details
    appointment_id = appointment_details[0].get('AppointmentId')
    patient_id = appointment_details[0].get('PatientId')
    from_date = appointment_details[0].get('FromDate')

    # Step 2: Cancel the current appointment
    cancel_appointment_url = "https://iochatbot.maximeyes.com/api/appointment/cancelappoinmentschatbot"
    cancel_payload = {
        "PatientId": patient_id,
        "AppointmentId": appointment_id,
        "from_date": from_date

    }
    cancel_response = requests.post(cancel_appointment_url, json=cancel_payload, headers=headers)

    if cancel_response.status_code != 200:
        return f"Failed to cancel the appointment. Status code: {cancel_response.status_code}"

    # Step 3: Update the appointment status
    update_status_url = f"https://iochatbot.maximeyes.com/api/appointment/updateappointmentstatus/{appointment_id}"
    update_status_payload = {
        "Status": "Cancelled"
    }
    update_status_response = requests.put(update_status_url, json=update_status_payload, headers=headers)

    if update_status_response.status_code != 200:
        return f"Failed to update appointment status. Status code: {update_status_response.status_code}"

    # Step 4: Get new slots for rescheduling
    open_slots_url = f"https://iochatbot.maximeyes.com/api/appointment/openslotforchatbot?fromDate={from_date}&isOpenSlotsOnly=true"
    open_slots_response = requests.get(open_slots_url, headers=headers)

    if open_slots_response.status_code != 200:
        return f"Failed to get open slots. Status code: {open_slots_response.status_code}"

    try:
        open_slots = open_slots_response.json()
    except ValueError:
        return "Failed to parse open slots response as JSON."

    if not open_slots:
        return "No open slots available for rescheduling."

    return open_slots

# Function to generate response using fine-tuned model
def generate_response(prompt, max_length=512, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response

# Main function to handle user query
def handle_user_query(user_query):
    # Identify the user's intent
    print("user_quesrt",user_query)
    intent = identify_intent(user_query)
    print(f"Identified intent: {intent}")

    # If the intent is to reschedule an appointment
    if "reschedule" in user_query:

        # Extract required information
        extracted_info = fetch_info(user_query)
        print(type(extracted_info))
        print(f"Extracted info: {extracted_info}")

        # Extract Lastname and DateOfBirth
        Lastname = extracted_info.get('Lastname')
        DateOfBirth = extracted_info.get('DateOfBirth')

        # Get authentication token
        auth_token = get_auth_token(vendor_id, vendor_password, account_id, account_password)
        if not auth_token:
            return "Failed to authenticate."

        # Reschedule the appointment and get open slots
        open_slots = reschedule_appointment(auth_token, Lastname, DateOfBirth)
        return open_slots

    # If the intent is not related to rescheduling, generate a response
    else:
        response = generate_response(user_query)
        return response
    
@app.route("/")
def index():

    return render_template("index.html")

# Function to generate response
def generate_response(prompt, max_length=512, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return response

# Handle POST requests to gpt2 endpoint
@app.route("/api/gpt2-chatbot", methods=["POST"])
def gpt2_chatbot():
    print("Received request:", request.json or request.data)
    if not request.is_json:
        return jsonify({"error": "Invalid content type, must be application/json"}), 400
    
    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON format, expected a JSON object"}), 400
        
        input_message = data["message"]

        print("Received message:", input_message)

        if not input_message:
            return jsonify({"error": "Missing 'message' in request data"}), 400

        response = handle_user_query(input_message)
        # response = generate_response(input_message)
        print("Generated response:", response[0])

        return jsonify({"response": response[0]})
    except KeyError:
        return jsonify({"error": "Missing 'message' in request data"}), 400
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

# Handle POST requests to dialo endpoint
@app.route("/api/dialo-chatbot", methods=["POST"])
def dialo_chatbot():
    input_message = request.json["message"]
    print("Received message:", input_message)
    
    response = chat(input_message, "microsoft/DialoGPT-medium")
    response = response.get("response", "")  # Extract the "response" value
    print("Generated response:", response)

    return jsonify({"response": response})

# Run the Flask application
if __name__ == "__main__":
    app.run(port=5000)
