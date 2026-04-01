import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
 
load_dotenv()
 
# ── Veritabanı ────────────────────────────────────────────────────────────────
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5442/postgres?gssencmode=disable",
)
 
# ── Memory ayarları ───────────────────────────────────────────────────────────
TRIM_THRESHOLD = 20   # Kaç mesaj birikince özetleme tetiklenir
KEEP_RECENT    = 4    # Özetlemeden sonra korunacak son mesaj sayısı
MAX_TOKENS     = 3000 # Trim güvenlik eşiği (kelime bazlı yaklaşık)
 
# ── Model ─────────────────────────────────────────────────────────────────────
model = ChatAnthropic(model="claude-haiku-4-5-20251001")