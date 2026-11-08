import os, queue, threading, requests
from dotenv import load_dotenv

load_dotenv("../.env")
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")
q = queue.Queue(maxsize=50)

def _worker():
    while True:
        text, image_path, silent = q.get()
        try:
            base = f"https://api.telegram.org/bot{TOKEN}"
            if image_path:
                with open(image_path, "rb") as f:
                    requests.post(f"{base}/sendPhoto", timeout=20,
                                  data={"chat_id": CHAT, "caption": text, "disable_notification": silent},
                                  files={"photo": f})
            else:
                requests.post(f"{base}/sendMessage", timeout=20,
                              data={"chat_id": CHAT, "text": text, "disable_notification": silent})
        except Exception as e:
            print("telegram error:", e)
        q.task_done()

threading.Thread(target=_worker, daemon=True).start()

def notify(text, image_path=None, silent=False):
    try:
        q.put_nowait((text, image_path, silent))
    except queue.Full:
        pass