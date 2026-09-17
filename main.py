import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types
import firebase_admin
from firebase_admin import credentials, db
import paho.mqtt.client as mqtt

app = FastAPI()

# 1. 初始化 Firebase (從 Render Secret Files 讀取私鑰)
CRED_PATH = "firebase-key.json" 
if os.path.exists(CRED_PATH):
    try:
        cred = credentials.Certificate(CRED_PATH)
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.environ.get("FIREBASE_DB_URL")
        })
        print("Firebase 初始化成功！")
    except Exception as e:
        print(f"Firebase 初始化失敗: {e}")

# 2. 初始化 Google GenAI SDK
api_key = os.environ.get("GEMINI_API_KEY")
ai_client = genai.Client(api_key=api_key) if api_key else None

# 3. 初始化 MQTT (加入防呆，避免變數漏掉時直接崩潰)
mqtt_client = mqtt.Client()
raw_host = os.environ.get("MQTT_HOST", "")
clean_host = raw_host.replace("mqtts://", "").replace("mqtt://", "").replace("https://", "").replace("http://", "").strip()

if clean_host:
    try:
        mqtt_client.tls_set()
        mqtt_client.username_pw_set(
            os.environ.get("MQTT_USER"), 
            os.environ.get("MQTT_PASSWORD")
        )
        mqtt_client.connect(clean_host, int(os.environ.get("MQTT_PORT", 8883)), 60)
        mqtt_client.loop_start()
        print("MQTT 連線成功！")
    except Exception as e:
        print(f"MQTT 連線失敗 (但不影響伺服器啟動): {e}")
else:
    print("警告: 尚未設定 MQTT_HOST 環境變數")

@app.post("/api/verify")
async def verify_door(file: UploadFile = File(...)):
    if not ai_client:
        raise HTTPException(status_code=500, detail="Gemini API Key 未設定")
    try:
        image_bytes = await file.read()
        prompt = "你是一個門禁系統 AI。請分析這張圖片，並決定是否允許開門。如果允許，請回覆 UNLOCK，否則回覆 LOCK。"
        
        response = ai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type or 'image/jpeg',
                ),
            ],
        )
        
        decision = response.text.strip().upper()
        
        if "UNLOCK" in decision:
            try:
                mqtt_client.publish("door", '{"action": "unlock"}')
            except:
                pass
            result_action = "unlock"
        else:
            try:
                mqtt_client.publish("lock", '{"action": "lock"}')
            except:
                pass
            result_action = "lock"
            
        return {"status": "success", "agent_decision": result_action}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "Agent Cloud Server is running smoothly!"}
