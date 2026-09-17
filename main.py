import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
import paho.mqtt.client as mqtt

app = FastAPI()

# 從 Render 的環境變數或安全檔案讀取設定
# ⚠️ 注意：Firebase 金鑰我們稍後會使用 Render 的「Secret Files」功能上傳
CRED_PATH = "firebase-key.json" 

if os.path.exists(CRED_PATH):
    cred = credentials.Certificate(CRED_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.environ.get("FIREBASE_DB_URL")
    })

# 初始化 Gemini 1.5 Flash
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# 初始化 MQTT (HiveMQ Cloud)
mqtt_client = mqtt.Client()
mqtt_client.tls_set()
mqtt_client.username_pw_set(
    os.environ.get("MQTT_USER"), 
    os.environ.get("MQTT_PASSWORD")
)
mqtt_client.connect(
    os.environ.get("MQTT_HOST"), 
    int(os.environ.get("MQTT_PORT", 8883)), 
    60
)
mqtt_client.loop_start()

@app.post("/api/verify")
async def verify_door(file: UploadFile = File(...)):
    try:
        # 1. 讀取 ESP32-CAM 傳來的圖片
        image_bytes = await file.read()
        image_blob = {'mime_type': file.content_type or 'image/jpeg', 'data': image_bytes}
        
        # 2. 呼叫 Gemini 1.5 Flash 進行分析與決策
        prompt = "你是一個門禁系統 AI。請分析這張圖片，並決定是否允許開門。如果允許，請回覆 UNLOCK，否則回覆 LOCK。"
        response = model.generate_content([prompt, image_blob])
        decision = response.text.strip().upper()
        
        # 3. 根據 Agent 判斷結果發送 MQTT 指令
        if "UNLOCK" in decision:
            mqtt_client.publish("door", '{"action": "unlock"}')
            result_action = "unlock"
        else:
            mqtt_client.publish("lock", '{"action": "lock"}')
            result_action = "lock"
            
        return {"status": "success", "agent_decision": result_action}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "Agent Cloud Server is running!"}
