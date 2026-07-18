# MyRouter — AI Sidecar

Sidecar **tương thích OpenAI API** cho [9Router], làm cầu nối tới ba backend:

| Backend | Thư viện / Giao thức | Model string |
|---|---|---|
| **Google Gemini** (web app, tài khoản Google Pro) | `gemini_webapi` | `gemini`, `gemini-3-flash`, `gemini-3-pro`, `gemini-3-flash-thinking` |
| **NotebookLM** | `notebooklm-py` | `<notebook_id>` (raw UUID của notebook) |
| **ComfyUI** (nhiều instance, tunnel, không auth) | HTTP API | tên instance trong bảng `comfy_instances` (vd `comfy1`) |

Điểm đặc biệt: **MySQL là nguồn chân lý cho Google auth** — đăng nhập một lần, cookie được lưu vào DB, tự materialize ra file khi cần, tự đồng bộ khi Google xoay cookie, và **tự re-login** khi hết hạn (thường im lặng trong ~5 giây nhờ phiên browser persistent).

---

## 1. Kiến trúc

```
app/
├── main.py            # FastAPI app, lifespan (migration, sync task), mount admin
├── config.py          # Settings (pydantic-settings, đọc .env)
├── db.py              # SQLAlchemy async engine + ensure_schema() (tự migrate)
├── models.py          # ApiKey, GoogleProfile, ComfyInstance, GeminiConversation, RequestLog
├── security.py        # Bearer auth 2 loại key + request logging
├── google_auth.py     # Cầu nối DB↔storage_state, login subprocess, auto re-login
├── pool.py            # Pool client NotebookLM/Gemini theo profile (lazy, retry khi expired)
├── comfy.py           # Workflow builder, info/queue, tải ảnh
├── routes/
│   ├── chat.py        # /v1/chat/completions + /v1/conversations
│   ├── images.py      # /v1/images/generations + /v1/comfy/*
│   ├── models_list.py # /v1/models
│   └── notebooklm.py  # /v1/notebooklm/* (generate, artifacts, status, download)
└── admin/             # Dashboard SQLAdmin (/admin) + API Playground
```

### Hai loại API key (bảng `api_keys`)

| `key_type` | Gắn với | Dùng được |
|---|---|---|
| `google` | một Google profile (`profile_name`) | Gemini **và** NotebookLM (chung cookie) |
| `comfy` | đúng một ComfyUI instance (`comfy_instance`) | các endpoint ảnh của instance đó |

Dùng chéo bị chặn 403. `/v1/models` trả danh sách theo loại key.

### Luồng Google auth

```
Login (dashboard/CLI) → storage_state.json → import vào DB (google_profiles)
Request đến → materialize DB→file → NotebookLMClient.from_storage(keepalive)
                                  → GeminiClient (full cookie jar từ DB)
Cookie xoay vòng → notebooklm-py ghi lại file → sync file→DB (định kỳ 10')
Hết hạn giữa request → tự chạy LOGIN_COMMAND → retry ngay trong request đó
```

Ba quy tắc quan trọng (đã trả giá để học được):
1. **Mỗi profile = đúng một tài khoản Google** trong phiên browser (gemini_webapi không hỗ trợ `authuser`). Dùng checkbox **fresh** khi đổi tài khoản.
2. **Chỉ một bên xoay cookie**: notebooklm-py keepalive. Gemini chạy `auto_refresh=False`, và cache cookie riêng của gemini_webapi bị xóa trước mỗi init (cache cũ làm phiên degraded → chat không hiện trong lịch sử Gemini web).
3. Không sửa tay `storage_state.json` — DB sẽ ghi đè/được ghi đè theo mtime.

---

## 2. Yêu cầu

- **Windows** có màn hình (login mở cửa sổ browser; Edge/Chrome/Chromium)
- **Python 3.12+**, **MySQL/MariaDB** chạy local
- Tài khoản Google đã dùng được Gemini + NotebookLM
- Các ComfyUI instance truy cập được qua HTTP(S)

## 3. Cài đặt & Deploy

```powershell
cd e:\SilverStar\python\MyRouter
python -m pip install -r requirements.txt

# MySQL: chỉ cần database tồn tại — bảng tự tạo/migrate khi khởi động
mysql -u root -e "CREATE DATABASE IF NOT EXISTS myrouter CHARACTER SET utf8mb4"

copy .env.example .env    # rồi sửa theo máy
```

`.env` tối thiểu:

```ini
DATABASE_URL=mysql+aiomysql://root:@localhost:3306/myrouter
ADMIN_USERNAME=admin
ADMIN_PASSWORD=<đặt-mật-khẩu>
SECRET_KEY=<chuỗi-ngẫu-nhiên-dài>
# Lệnh login theo máy (--storage và NOTEBOOKLM_PROFILE tự thêm)
LOGIN_COMMAND=notebooklm login --browser msedge
```

Các knob khác (đều có mặc định hợp lý): `COMFY_CHECKPOINT`, `COMFY_DEFAULT_SIZE`, `COMFY_STEPS/CFG/NEGATIVE_PROMPT`, `COMFY_POLL_INTERVAL/TIMEOUT`, `AUTO_RELOGIN` (mặc định bật), `AUTO_RELOGIN_WAIT`, `PROFILE_SYNC_INTERVAL`, `LOGIN_TIMEOUT`, `NOTEBOOK_KEEPALIVE`, `LOG_LEVEL`.

Chạy server:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Thiết lập lần đầu (qua dashboard `http://localhost:8000/admin`)

1. **Status** → *Add profile & login* (tick **fresh**) → đăng nhập Google trong cửa sổ browser → auth tự lưu vào DB.
2. **ComfyUI Instances** → thêm từng instance (name = model string, base_url).
3. **API Keys** → tạo key `google` (chọn profile) và key `comfy` cho **từng** instance.
4. **API Playground** → test ngay: chọn key → model tự load → chat/sinh ảnh.

Swagger có nút Authorize tại `/docs`. Health check: `/healthz`.

---

## 4. Dùng API

Chuẩn OpenAI — trỏ `base_url` về sidecar là xong:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="<google-key>")

# Gemini theo model (stateless)
r = client.chat.completions.create(model="gemini-3-pro",
        messages=[{"role": "user", "content": "Xin chào"}])

# NotebookLM: notebook id làm model
r = client.chat.completions.create(model="e5bad695-7a4e-431b-aee0-83800eca4edf",
        messages=[{"role": "user", "content": "Notebook này nói về gì?"}])
```

**Gemini conversation** (đa lượt server-side, mở rộng ngoài chuẩn OpenAI):

```jsonc
POST /v1/chat/completions
{ "model": "gemini-3-pro", "conversation_id": "new",   // "new" tạo mới
  "messages": [{ "role": "user", "content": "Tôi tên Quyền" }] }
// → response có "conversation_id": "conv-..." — gửi lại id đó để chat tiếp
```

Quản lý: `GET /v1/conversations`, `DELETE /v1/conversations/{id}`.

**NotebookLM commands** (google key):

```
POST /v1/notebooklm/generate                {notebook_id, type, instructions?}
     # type: audio|video|report|study_guide|quiz|flashcards|slide_deck|infographic|data_table|mind_map
GET  /v1/notebooklm/{nb}/artifacts[?type=]
GET  /v1/notebooklm/{nb}/status/{task_id}
GET  /v1/notebooklm/{nb}/download/{type}[?artifact_id=&format=]   # trả file (pdf/pptx/md/csv/json/png/mp4…)
```

**Ảnh — ComfyUI** (comfy key; instance suy từ key):

```jsonc
POST /v1/images/generations
{ "prompt": "a cute cat wearing a hat",
  "size": "auto",                    // hoặc "512x512"… ("auto" → COMFY_DEFAULT_SIZE)
  "response_format": "b64_json",     // url (mặc định) | b64_json | binary
  // knob tùy chọn:
  "checkpoint": "...", "sampler": "...", "scheduler": "...",
  "steps": 25, "cfg": 7.0, "seed": 42, "negative_prompt": "...",
  "workflow": { }                    // hoặc gửi nguyên workflow ComfyUI (placeholder {prompt}{seed}{width}{height})
}
GET /v1/comfy/info    # checkpoint/sampler/scheduler thật của instance
GET /v1/comfy/queue   # độ sâu hàng đợi
```

**Tích hợp 9Router**: mỗi provider = `base_url` sidecar + key tương ứng (google key cho chat, comfy key riêng từng instance cho ảnh; Output Format "JSON (Base64)"/"Binary File" và Size "auto" được hỗ trợ sẵn).

---

## 5. Vận hành & xử lý sự cố

| Hiện tượng | Nguyên nhân / Cách xử lý |
|---|---|
| 502 `profile_auth_expired` | Auth hết hạn và auto re-login chưa xong — nếu có cửa sổ browser đang mở trên máy chủ, đăng nhập nốt rồi gọi lại; hoặc vào Status bấm Re-login. Cooldown 2 phút giữa các lần tự thử. |
| Cửa sổ Edge tự bật khi gọi API | Là auto re-login (thường tự đóng ~5s). Tắt bằng `AUTO_RELOGIN=false`. |
| 403 `wrong_key_type` | Dùng nhầm loại key (google ↔ comfy). |
| Chat Gemini không hiện trong lịch sử web | Đã fix (cache cookie degraded); nếu tái diễn: Re-login profile, kiểm tra log có `UNAUTHENTICATED`. |
| `model_not_found` khi gọi Gemini | Xem model hợp lệ qua `GET /v1/models`. |
| ComfyUI 502/504 | Tunnel down hoặc checkpoint không tồn tại — kiểm tra `GET /v1/comfy/info`, `/v1/comfy/queue`. |
| Theo dõi | Bảng **Request Logs** trên dashboard (endpoint, status, latency, error); trang **Status** có thống kê 24h + reachability. |

**Bảo mật**: DB chứa API key và cookie Google (bảng `google_profiles.storage_state`) — chỉ chạy trên máy tin cậy, đặt mật khẩu MySQL/admin thật, không expose `/admin` ra ngoài mạng nội bộ.
