"""
Agent Gate Server
==================
流程：
  MQTT 訂閱 "car" (數字，從伺服器啟動時的第一筆訊息校準基準值)
    -> 數字變化 -> 到 Google Drive 讀取最新照片
    -> Gemini 只負責「辨識車牌文字」(結構化 JSON 輸出，不做開/關門決策)
    -> 用你自己的程式碼邏輯去比對 Firebase authorized_vehicles 白名單
    -> 決策結果 (deterministic) 決定 MQTT publish unlock / lock
    -> 寫入 Firebase door_access_logs 留稽核紀錄

安全設計要點：
  - AI 只做「感知」(圖片 -> 文字)，不做「決策」(開不開門)，避免 prompt injection
    直接控制實體鎖。
  - 白名單比對、開鎖與否，全部是確定性 (deterministic) 的 Python if/else。
  - 每一次事件都留下稽核紀錄 (照片來源 file_id、辨識結果、confidence、最終決策)。
  - /api/manual-verify 這種對外 endpoint 一律要求 API Key。
"""

import os
import re
import json
import logging
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from google import genai
from google.genai import types

import firebase_admin
from firebase_admin import credentials, db

import paho.mqtt.client as mqtt

from googleapiclient.discovery import build
from google.oauth2 import service_account
import io
from googleapiclient.http import MediaIoBaseDownload

import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("gate-agent")
# 有些部署環境（如 docker/雲端 log 收集器）會把 stdout buffer 起來，
# 導致 log 延遲很久才出現。這裡強制 unbuffered，讓你能即時看到流程進度。
for h in log.handlers or logging.getLogger().handlers:
    h.flush()

app = FastAPI()

# 背景執行緒池，避免在 MQTT callback thread 裡長時間阻塞
executor = ThreadPoolExecutor(max_workers=2)

# ---------------------------------------------------------------------------
# 1. Firebase 初始化
# ---------------------------------------------------------------------------
CRED_PATH = os.environ.get("FIREBASE_CRED_PATH", "firebase-key.json")
firebase_ready = False
if os.path.exists(CRED_PATH):
    try:
        cred = credentials.Certificate(CRED_PATH)
        firebase_admin.initialize_app(cred, {
            "databaseURL": os.environ.get("FIREBASE_DB_URL")
        })
        firebase_ready = True
        log.info("Firebase 初始化成功")
    except Exception as e:
        log.error(f"Firebase 初始化失敗: {e}")
else:
    log.warning(f"找不到 Firebase 憑證檔案: {CRED_PATH}")

# ---------------------------------------------------------------------------
# 2. Gemini 初始化
# ---------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
ai_client = genai.Client(api_key=api_key) if api_key else None

PLATE_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# 要求 Gemini 只回傳結構化 JSON：是否看得到車牌、車牌文字、信心分數
PLATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "plate_visible": {"type": "boolean"},
        "plate_number": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["plate_visible", "plate_number", "confidence"],
}

PLATE_PROMPT = (
    "你是一個車牌辨識系統。請仔細觀察這張照片中的車輛車牌。\n"
    "只回傳 JSON，不要做任何「是否開門」的判斷，那不是你的工作。\n"
    "欄位說明：\n"
    "- plate_visible: 是否能清楚辨識出車牌 (boolean)\n"
    "- plate_number: 辨識出的車牌文字，若看不到請回傳空字串\n"
    "- confidence: 你對這次辨識結果的信心分數 (0.0 ~ 1.0)"
)

MIN_CONFIDENCE = float(os.environ.get("PLATE_MIN_CONFIDENCE", "0.75"))


def extract_plate_number(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """呼叫 Gemini，只做「圖片 -> 車牌文字」的感知工作，不做開門決策。"""
    if not ai_client:
        raise RuntimeError("Gemini API Key 未設定")

    log.info(f"🤖 Gemini 解析中...（model={PLATE_MODEL}, 圖片大小={len(image_bytes)} bytes）")

    response = ai_client.models.generate_content(
        model=PLATE_MODEL,
        contents=[
            PLATE_PROMPT,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PLATE_RESPONSE_SCHEMA,
            temperature=0,
        ),
    )
    data = json.loads(response.text)
    result = {
        "plate_visible": bool(data.get("plate_visible", False)),
        "plate_number": normalize_plate(data.get("plate_number", "")),
        "confidence": float(data.get("confidence", 0.0)),
    }

    if result["plate_visible"] and result["plate_number"]:
        log.info(f"✅ Gemini 辨識完成：車牌={result['plate_number']} 信心度={result['confidence']:.2f}")
    else:
        log.info(f"⚠️ Gemini 未能辨識出車牌（信心度={result['confidence']:.2f}）")

    return result


def normalize_plate(raw: str) -> str:
    """車牌正規化：轉大寫、只保留英數字與 '-'，去除多餘空白。
    Firebase Realtime Database 的 key 不可包含 . # $ [ ] /，'-' 沒問題。
    """
    if not raw:
        return ""
    cleaned = raw.strip().upper()
    cleaned = re.sub(r"[^A-Z0-9\-]", "", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# 3. Google Drive 初始化 (讀取最新照片)
# ---------------------------------------------------------------------------
DRIVE_CRED_PATH = os.environ.get("DRIVE_CRED_PATH", "drive-key.json")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

drive_service = None
if os.path.exists(DRIVE_CRED_PATH):
    try:
        drive_creds = service_account.Credentials.from_service_account_file(
            DRIVE_CRED_PATH, scopes=DRIVE_SCOPES
        )
        drive_service = build("drive", "v3", credentials=drive_creds)
        log.info("Google Drive 初始化成功")
    except Exception as e:
        log.error(f"Google Drive 初始化失敗: {e}")
else:
    log.warning(f"找不到 Drive 憑證檔案: {DRIVE_CRED_PATH}")


def fetch_latest_photo() -> tuple[bytes, str, str]:
    """從指定資料夾抓「最新建立」的一張照片，回傳 (bytes, mime_type, file_id)。"""
    if not drive_service:
        raise RuntimeError("Google Drive 尚未初始化")
    if not DRIVE_FOLDER_ID:
        raise RuntimeError("未設定 DRIVE_FOLDER_ID")

    log.info("📷 正在從 Google Drive 讀取最新照片...")

    results = drive_service.files().list(
        q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false",
        orderBy="createdTime desc",
        pageSize=1,
        fields="files(id, name, mimeType, createdTime)",
    ).execute()

    files = results.get("files", [])
    if not files:
        raise RuntimeError("資料夾中沒有找到照片")

    file_info = files[0]
    file_id = file_info["id"]
    mime_type = file_info.get("mimeType", "image/jpeg")

    request = drive_service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    log.info(f"📷 照片下載完成：{file_info.get('name')} (file_id={file_id})")

    return buf.getvalue(), mime_type, file_id


# ---------------------------------------------------------------------------
# 4. Firebase 白名單比對 + 紀錄寫入 (確定性邏輯，不靠 AI)
# ---------------------------------------------------------------------------
def check_authorized(plate: str) -> dict | None:
    """回傳 authorized_vehicles/{plate} 的資料，不存在則回傳 None。"""
    if not firebase_ready:
        raise RuntimeError("Firebase 尚未初始化")
    log.info(f"🔍 查詢白名單中：authorized_vehicles/{plate}")
    ref = db.reference(f"authorized_vehicles/{plate}")
    record = ref.get()
    log.info(f"🔍 白名單查詢結果：{'找到' if record else '不存在'}")
    return record


def decide_action(plate_result: dict) -> tuple[str, str, str | None, dict]:
    """
    根據辨識結果 + 白名單資料，決定最終動作。全部是確定性規則：
      1. 沒看到車牌 / 信心太低 -> LOCK, NO_PLATE_DETECTED
      2. 車牌不在白名單        -> LOCK, DENIED
      3. 白名單但 status != active（或已過期）-> LOCK, DENIED
      4. 白名單且有效          -> UNLOCK, SUCCESS
    回傳 (action, result, user_id, extra_log_fields)
    """
    plate = plate_result["plate_number"]

    if not plate_result["plate_visible"] or not plate or plate_result["confidence"] < MIN_CONFIDENCE:
        return "LOCK", "NO_PLATE_DETECTED", None, {"plate_number": plate or None}

    record = check_authorized(plate)
    if not record:
        return "LOCK", "DENIED", None, {"plate_number": plate}

    status = record.get("status", "active")  # 沒有 status 欄位就當作 active，向下相容
    valid_until = record.get("valid_until")
    if valid_until:
        try:
            if datetime.fromisoformat(valid_until.replace("Z", "+00:00")) < datetime.now(timezone.utc):
                return "LOCK", "DENIED", record.get("user_id"), {"plate_number": plate, "reason": "expired"}
        except ValueError:
            log.warning(f"valid_until 格式無法解析: {valid_until}")

    if status != "active":
        return "LOCK", "DENIED", record.get("user_id"), {"plate_number": plate, "reason": f"status={status}"}

    return "UNLOCK", "SUCCESS", record.get("user_id"), {"plate_number": plate}


def write_access_log(action: str, result: str, user_id: str | None, extra: dict, file_id: str | None = None):
    if not firebase_ready:
        log.error("Firebase 未初始化，無法寫入 log")
        return
    entry = {
        "action": action,
        "result": result,
        "user_id": user_id or "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    entry.update(extra)
    if file_id:
        entry["source_photo_id"] = file_id

    # 用 push() 產生不會撞號的唯一 key，避免多個事件同時發生時互相覆蓋
    db.reference("door_access_logs").push(entry)
    log.info(f"寫入 access log: {entry}")


# ---------------------------------------------------------------------------
# 5. MQTT：發布開關門指令 + 訂閱 "car" 觸發訊號
# ---------------------------------------------------------------------------
DOOR_TOPIC = os.environ.get("MQTT_DOOR_TOPIC", "door_control")
CAR_TOPIC = os.environ.get("MQTT_CAR_TOPIC", "car")

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

raw_host = os.environ.get("MQTT_HOST", "")
clean_host = (
    raw_host.replace("mqtts://", "")
    .replace("mqtt://", "")
    .replace("https://", "")
    .replace("http://", "")
    .strip()
)

# 用來記錄目前已知的 car 計數值。None 代表「還沒校準」，
# 收到的第一筆訊息只拿來設定基準值，不觸發流程（避免 retained message 造成誤觸發）。
_last_car_count_lock = threading.Lock()
_last_car_count: int | None = None


def process_gate_event():
    """完整流程：抓照片 -> 辨識車牌 -> 比對白名單 -> 開關門 -> 寫 log。
    這個函式跑在背景執行緒中，允許同步阻塞呼叫。
    """
    log.info("=" * 60)
    log.info("🚗 偵測到車輛，開始門禁判斷流程...")

    file_id = None
    try:
        image_bytes, mime_type, file_id = fetch_latest_photo()

        plate_result = extract_plate_number(image_bytes, mime_type)

        log.info("⚖️ 比對白名單、計算最終決策中...")
        action, result, user_id, extra = decide_action(plate_result)

        emoji = "🟢" if action == "UNLOCK" else "🔴"
        log.info(f"{emoji} 決策結果：action={action}  result={result}  user_id={user_id}")

    except Exception as e:
        log.exception("❌ gate event 處理失敗，安全起見預設鎖門")
        action, result, user_id, extra = "LOCK", "ERROR", None, {"error": str(e)}

    # 無論成功失敗，預設安全動作是 LOCK；只有明確 SUCCESS 才 UNLOCK
    try:
        payload = json.dumps({"action": action.lower()})
        log.info(f"📡 傳送 MQTT 指令中... topic={DOOR_TOPIC} payload={payload}")
        mqtt_client.publish(DOOR_TOPIC, payload)
        log.info("📡 MQTT 指令已送出")
    except Exception as e:
        log.error(f"📡 MQTT publish 失敗: {e}")

    try:
        log.info("📝 寫入 Firebase door_access_logs 中...")
        write_access_log(action, result, user_id, extra, file_id=file_id)
        log.info("📝 Log 寫入完成")
    except Exception as e:
        log.error(f"📝 寫入 log 失敗: {e}")

    log.info("✅ 本次流程結束，繼續監聽 car topic...")
    log.info("=" * 60)


def on_car_message(client, userdata, msg):
    global _last_car_count
    try:
        value = int(msg.payload.decode().strip())
    except ValueError:
        log.warning(f"car topic 收到非數字訊息: {msg.payload!r}")
        return

    log.info(f"📨 收到 car topic 訊息，目前數值={value}")

    with _last_car_count_lock:
        if _last_car_count is None:
            # 第一筆訊息只用來校準基準值，不觸發（可能是 broker 的 retained message）
            _last_car_count = value
            log.info(f"car 計數器初始化為 {value}（不觸發流程）")
            return

        if value != _last_car_count:
            log.info(f"car 計數變化 {_last_car_count} -> {value}，觸發門禁流程")
            _last_car_count = value
            executor.submit(process_gate_event)


def on_connect(client, userdata, flags, reason_code, properties=None):
    log.info(f"MQTT 已連線，reason_code={reason_code}")
    client.subscribe(CAR_TOPIC)
    log.info(f"已訂閱 topic: {CAR_TOPIC}")


if clean_host:
    try:
        mqtt_client.tls_set()
        mqtt_client.username_pw_set(
            os.environ.get("MQTT_USER"),
            os.environ.get("MQTT_PASSWORD"),
        )
        mqtt_client.on_connect = on_connect
        mqtt_client.message_callback_add(CAR_TOPIC, on_car_message)
        mqtt_client.connect(clean_host, int(os.environ.get("MQTT_PORT", 8883)), 60)
        mqtt_client.loop_start()
        log.info("MQTT 連線成功")
    except Exception as e:
        log.error(f"MQTT 連線失敗 (但不影響伺服器啟動): {e}")
else:
    log.warning("未設定 MQTT_HOST，MQTT 功能停用")


# ---------------------------------------------------------------------------
# 6. 手動測試用 API（需要 API Key，方便你在沒有實體相機/MQTT 訊號時測試）
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("AGENT_API_KEY")


def require_api_key(x_api_key: str = Header(default=None)):
    if not API_KEY:
        # 沒設定 API_KEY 的話，代表你還在本機測試；正式部署務必設定這個環境變數
        log.warning("AGENT_API_KEY 未設定，/api/manual-verify 目前沒有驗證！")
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="無效的 API Key")


@app.post("/api/manual-verify", dependencies=[Depends(require_api_key)])
async def manual_verify(file: UploadFile = File(...)):
    """手動上傳照片測試辨識+決策流程，不透過 MQTT/Drive 觸發，方便除錯。"""
    if not ai_client:
        raise HTTPException(status_code=500, detail="Gemini API Key 未設定")

    image_bytes = await file.read()
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="檔案過大（上限 8MB）")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="只接受圖片檔案")

    try:
        plate_result = extract_plate_number(image_bytes, file.content_type or "image/jpeg")
        action, result, user_id, extra = decide_action(plate_result)
        write_access_log(action, result, user_id, extra, file_id="manual-upload")
        return {
            "status": "success",
            "plate_result": plate_result,
            "action": action.lower(),
            "result": result,
        }
    except Exception:
        log.exception("manual_verify 失敗")
        raise HTTPException(status_code=500, detail="伺服器內部錯誤，請查看伺服器日誌")


@app.post("/api/trigger", dependencies=[Depends(require_api_key)])
async def manual_trigger():
    """手動觸發一次完整流程（等同於收到 car topic 數字變化）。"""
    executor.submit(process_gate_event)
    return {"status": "triggered"}


@app.get("/")
def health_check():
    return {
        "status": "Agent Gate Server is running",
        "firebase_ready": firebase_ready,
        "drive_ready": drive_service is not None,
        "mqtt_connected": mqtt_client.is_connected() if clean_host else False,
    }
