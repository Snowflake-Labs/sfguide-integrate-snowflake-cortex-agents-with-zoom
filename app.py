import os
import requests
import snowflake.connector
from dotenv import load_dotenv
from flask import Flask, request, jsonify, redirect
import cortex_chat
import pandas as pd
import base64
import json

load_dotenv()

# Environment variables
ACCOUNT = os.getenv("ACCOUNT")
USER = os.getenv("DEMO_USER")
RSA_PRIVATE_KEY_PATH = os.getenv("RSA_PRIVATE_KEY_PATH")
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT")
SUPPORT_SEMANTIC_MODEL = os.getenv("SUPPORT_SEMANTIC_MODEL")
SUPPLY_CHAIN_SEMANTIC_MODEL = os.getenv("SUPPLY_CHAIN_SEMANTIC_MODEL")
VEHICLE_SEARCH_SERVICE = os.getenv("VEHICLE_SEARCH_SERVICE")
SEMANTIC_MODELS = [SUPPORT_SEMANTIC_MODEL, SUPPLY_CHAIN_SEMANTIC_MODEL]
SEARCH_SERVICES = [VEHICLE_SEARCH_SERVICE]
MODEL = os.getenv("MODEL")

ZOOM_ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")
ZOOM_CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_TOKEN_URL = os.getenv("ZOOM_TOKEN_URL")
ZOOM_CHAT_URL = os.getenv("ZOOM_CHAT_URL")
ZOOM_BOT_JID = os.getenv("ZOOM_BOT_JID")
ZOOM_REDIRECT_URI = os.getenv("ZOOM_REDIRECT_URI")

CONN = None
CORTEX_APP = None
DEBUG = False

app = Flask(__name__)

# Get Zoom access token
def get_access_token():
    auth_string = f"{ZOOM_CLIENT_ID}:{ZOOM_CLIENT_SECRET}"
    b64_auth = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {b64_auth}"
    }

    response = requests.post(
        f"{ZOOM_TOKEN_URL}?grant_type=client_credentials",
        headers=headers
    )

    if response.status_code != 200:
        print(response)
        response.raise_for_status()

    return response.json()['access_token']

def send_chat_message(user_jid, to_jid, message):
    token = get_access_token()
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    content = {
        "head": {
            "type": "message",
            "text": "Snowflake Cortex AI Response"
        },
        "body": [
            {
                "type": "message",
                "text": message
            }
        ]
    }

    payload = {
        "robot_jid": ZOOM_BOT_JID,
        "to_jid": to_jid,
        "user_jid": user_jid,
        "content": content  
    }

    response = requests.post(ZOOM_CHAT_URL, headers=headers, json=payload)

    if DEBUG:
        print("Zoom payload:")
        print(json.dumps(payload, indent=2))

    if response.status_code not in [200,201]:
        print(f"Zoom response: {response.status_code} {response.text}")
        response.raise_for_status()

    return response.status_code, response.json()

@app.route('/', methods=['GET'])
def oauth_redirect():
    code = request.args.get('code')
    if not code:
        return "Missing code", 400

    # Exchange code for token
    token_response = requests.post(
        ZOOM_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": ZOOM_REDIRECT_URI
        },
        auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    return redirect(f"https://zoom.us/launch/chat?jid={ZOOM_BOT_JID}")

# Route that Zoom Chat will POST to
@app.route('/askcortex', methods=['POST'])
def zoom_chat():
    data = request.get_json()
    user_jid = data.get('payload', {}).get('userJid')
    to_jid = data.get('payload', {}).get('toJid')
    prompt = data.get('payload', {}).get('cmd', '').strip()

    # sample request payload
    # {'event': 'bot_notification', 'payload': {'accountId': '2jy0Dr6nQTyxxxxxxxxxxxx', 'channelName': 'DZoom', 'cmd': 'hello', 
    # 'robotJid': 'v1fsgsvqogqXXXXXXXXXX@xmpp.zoom.us', 'timestamp': 1745344684085, 
    # 'toJid': '_e-fkvtft9XXXXXXXXXX@xmpp.zoom.us', 'triggerId': 'DbU0DiotTxxxxxxxx', 
    # 'userId': '_E-FKvTfT9KLXXXXXXXXXXXX', 
    # 'userJid': '_e-fkvtft9ZZZZZZZZZZZZZ@xmpp.zoom.us', 
    # 'userMemberId': '2vnF6Il-Usco8EWen5lLs1vhi1l8FYqTK8jgp-XXXXXXXXXXX', 'userName': 'Dash D', 'userStatus': 'authorized'}}

    agent_response = parse_agent_response(CORTEX_APP.chat(prompt))

    # Send Snowflake Cortex response to Zoom Chat
    send_status, zoom_response = send_chat_message(user_jid, to_jid, agent_response)
    return jsonify({"status": send_status, "response": zoom_response})

def parse_agent_response(content):
    try:
        if DEBUG:
            print(content)

        if content.get('sql'):
            sql = content['sql']
            interpreted_question = content['text']
            df = pd.read_sql(sql, CONN)
            data2answer = CORTEX_APP.data_to_answer(df.to_string())
            if DEBUG:
                print(df.to_string())
                print(f"{interpreted_question} \n\nAnswer: {data2answer['text']}")
            return data2answer['text']
        elif content.get('text'):
            answer = content['text']
            citations = content.get('citations','N/A')
            return answer + "\n\n* Citation(s): " + citations
        else:
            answer = 'Sorry, no response available! Please try asking another question.'
            return answer         
    except Exception as e:
        print(f"Error processing agent response: {str(e)}")
        return f"Sorry, encountered an error: {str(e)}."

# Initialize Snowflake and Cortex
def init():
    conn = snowflake.connector.connect(
        user=USER,
        authenticator="SNOWFLAKE_JWT",
        private_key_file=RSA_PRIVATE_KEY_PATH,
        account=ACCOUNT
    )

    if not conn.rest.token:
        print("[ERROR] Snowflake connection failed.")
        exit

    cortex_app = cortex_chat.CortexChat(
        AGENT_ENDPOINT,
        SEARCH_SERVICES,
        SEMANTIC_MODELS,
        MODEL,
        ACCOUNT,
        USER,
        RSA_PRIVATE_KEY_PATH
    )

    print("[INFO] Initialization complete")
    return conn, cortex_app

if __name__ == '__main__':
    CONN,CORTEX_APP = init()
    app.run(debug=True, port=5000)
