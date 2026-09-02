# MyRouter — AI Sidecar

Sidecar **tương thích OpenAI API** cho [9Router], làm cầu nối tới bốn backend:

| Backend | Thư viện / Giao thức | Model string |
|---|---|---|
| **Google Gemini** (web app, tài khoản Google Pro) | `gemini_webapi` | `gemini`, `gemini-3-flash`, `gemini-3-pro`, `gemini-3-flash-thinking` |
| **NotebookLM** | `notebooklm-py` | `<notebook_id>` (raw UUID của notebook) |
| **ComfyUI** (nhiều instance, tunnel, không auth) | HTTP API | tên instance trong bảng `comfy_instances` (vd `comfy1`) |
| **Microsoft Copilot** (nhiều tài khoản MS) | `Windows-Copilot-API` (vendored, `third_party/`) | `copilot` (một model duy nhất) |

Điểm đặc biệt: **MySQL là nguồn chân lý cho Google auth** — đăng nhập một lần, cookie được lưu vào DB, tự materialize ra file khi cần, tự đồng bộ khi Google xoay cookie, và **tự re-login** khi hết hạn (thường im lặng trong ~5 giây nhờ phiên browser persistent).

---

## 1. Kiến trúc

```
app/
├── main.py            # FastAPI app, lifespan (migration, sync task), mount admin
├── config.py          # Settings (pydantic-settings, đọc .env)
├── db.py              # SQLAlchemy async engine + ensure_schema() (tự migrate)
├── models.py          # ApiKey, Google/CopilotProfile, ComfyInstance, Gemini/CopilotConversation, RequestLog
├── security.py        # Bearer auth 3 loại key (google/comfy/copilot) + request logging
├── google_auth.py     # Cầu nối DB↔storage_state, login subprocess, auto re-login
├── copilot_lib.py     # Bọc thư viện vendored: SessionCopilotClient (session_dir mỗi account, cho http mode)
├── copilot_browser_chat.py  # Chat qua headless browser (mặc định) — đọc frame appendText từ chat WS
├── copilot_auth.py    # Session dir + status + login subprocess (scripts/copilot_login.py)
├── copilot_pool.py    # Pool Copilot, khóa serialize mỗi account, chọn browser|http mode
├── pool.py            # Pool client NotebookLM/Gemini theo profile (lazy, retry khi expired)
├── comfy.py           # Workflow builder, info/queue, tải ảnh, upload/analyze input, ephemeral
├── comfy_provision.py # ComfyUI-Manager V3.41: dò node/model thiếu, cài, poll status
├── routes/
│   ├── chat.py        # /v1/chat/completions + /v1/conversations (Gemini + NotebookLM)
│   ├── images.py      # /v1/images/generations + /v1/comfy/*
│   ├── copilot_api.py # /copilot/v1/* (models, chat, conversations)
│   ├── gemini_api.py notebooklm_api.py comfy_api.py  # 3 provider surface cho 9Router
│   ├── models_list.py # /v1/models
│   └── notebooklm.py  # /v1/notebooklm/* (generate, artifacts, status, download)
├── admin/             # Dashboard SQLAdmin (/admin) + API Playground
scripts/copilot_login.py            # Đăng nhập Copilot (Playwright) vào session dir mỗi account
third_party/windows_copilot_api/    # thư viện Copilot vendored (MIT) — có sửa cục bộ (xem header useragent.py)
```

### Ba loại API key (bảng `api_keys`)

| `key_type` | Gắn với | Dùng được |
|---|---|---|
| `google` | một Google profile (`profile_name`) | Gemini **và** NotebookLM (chung cookie) |
| `comfy` | đúng một ComfyUI instance (`comfy_instance`) | các endpoint ảnh của instance đó |
| `copilot` | một Copilot profile (`copilot_profile`) | `/copilot/v1/*` của account đó |

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
2. **Chỉ một bên xoay cookie**: notebooklm-py keepalive. Gemini chạy `auto_refresh=False`, và cache cookie riêng của gemini_webapi bị xóa trước mỗi init bằng `clear_cookies_cache()` (cache cũ làm phiên degraded → chat không hiện trong lịch sử Gemini web).
3. Không sửa tay `storage_state.json` — DB sẽ ghi đè/được ghi đè theo mtime.

---

## 2. Yêu cầu

**Bắt buộc:** **Python 3.12+** và **MySQL/MariaDB** local.

**Theo backend bạn dùng** (chỉ cần cái nào bạn bật):

| Backend | Cần thêm |
|---|---|
| Gemini / NotebookLM | Tài khoản Google (đã dùng được Gemini + NotebookLM) + máy **có màn hình** để login (cửa sổ browser Edge/Chrome/Chromium). |
| ComfyUI (ảnh) | Ít nhất một ComfyUI instance truy cập được qua HTTP(S). |
| Copilot | `playwright install chromium` một lần + máy **có màn hình** để login; chat chạy qua headless browser. |

### Phiên bản đã ghim (đọc trước khi nâng cấp)

`notebooklm-py` và `gemini-webapi` là client reverse-engineered, đổi API surface
rất nhanh, nên `requirements.txt` **ghim chính xác** thay vì `>=`:

| Gói | Pin | Vì sao |
|---|---|---|
| `notebooklm-py` | `==0.8.1` | 0.8.0 đổi error contract ("absence and refusal raise") — `app/pool.py` bắt `AuthError` có kiểu vì việc này, và `generate_study_guide` đã đổi tham số free-text sang `extra_instructions`. |
| `gemini-webapi` | `==2.1.1` | 2.1.0 sửa bug ChatSession dùng chung `DEFAULT_METADATA` (trước đây mọi hội thoại bị gộp làm một). 2.1.x cũng đổi tên `ModelInvalid`/`TemporarilyBlocked`/`UsageLimitExceeded` → hậu tố `*Error`. |
| `curl_cffi` | `~=0.16.2` | Bắt buộc bởi gemini-webapi 2.1.1; cũng là bản đầu tiên có wheel Python 3.14 và target impersonate `chrome150`. |
| `playwright` | `>=1.62` | Bundle Chromium 151. `third_party/.../useragent.py` đọc `browsers.json` của Playwright để dựng UA cho Copilot — nếu file/schema đó đổi chỗ, UA lặng lẽ rơi về fallback và Cloudflare clearance hỏng. |

Khi bump: chạy lại `playwright install chromium`, và kiểm tra
`IMPERSONATE_TARGET` trong `third_party/windows_copilot_api/copilot/useragent.py`
vẫn là target chrome mới nhất mà curl_cffi hỗ trợ.

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

Các knob khác (đều có mặc định hợp lý): `COMFY_CHECKPOINT`, `COMFY_DEFAULT_SIZE`, `COMFY_STEPS/CFG/NEGATIVE_PROMPT`, `COMFY_POLL_INTERVAL/TIMEOUT`, `COMFY_AUTO_PROVISION`, `COMFY_PROVISION_TIMEOUT`, `COMFY_EPHEMERAL`, `CHAT_TEMPORARY` (ephemeral chat mặc định), `COPILOT_CHAT_MODE` (browser|http, mặc định browser), `COPILOT_BROWSER_HEADLESS`, `COPILOT_BROWSER_CHAT_TIMEOUT`, `COPILOT_SESSION_ROOT`, `COPILOT_INTERACTIVE_CLEAR`/`COPILOT_HEADLESS_CLEAR` (chỉ dùng cho http mode), `AUTO_RELOGIN` (mặc định bật), `AUTO_RELOGIN_WAIT`, `PROFILE_SYNC_INTERVAL`, `LOGIN_TIMEOUT`, `NOTEBOOK_KEEPALIVE`, `LOG_LEVEL`.

Chạy server:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Thiết lập lần đầu (qua dashboard `http://localhost:8000/admin`)

1. **Status** → *Add profile & login* (tick **fresh**) → đăng nhập Google trong cửa sổ browser → auth tự lưu vào DB.
2. **Status → Copilot Profiles** (tuỳ chọn) → *Add account & login* → đăng nhập Microsoft/Google; nếu là Google, gửi 1 tin nhắn trong cửa sổ để hoàn tất.
3. **ComfyUI Instances** → thêm từng instance (name = model string, base_url).
4. **API Keys** → tạo key `google` (chọn profile), `comfy` (từng instance), và `copilot` (chọn Copilot profile).
5. **API Playground** → test ngay: chọn key → model tự load → chat/sinh ảnh.

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
  "workflow": { },                   // hoặc gửi nguyên workflow ComfyUI (placeholder {prompt}{seed}{width}{height})
  "auto_provision": true,            // cài node/model còn thiếu của workflow trước khi render (mặc định COMFY_AUTO_PROVISION)
  "ephemeral": true                  // không lưu job trên ComfyUI (mặc định COMFY_EPHEMERAL)
}
GET /v1/comfy/info    # checkpoint/sampler/scheduler thật của instance
GET /v1/comfy/queue   # độ sâu hàng đợi
```

**Tự cài đặt instance (ComfyUI-Manager V3.41)** — đẩy một workflow, Manager cài custom node còn thiếu và tải model/embedding còn thiếu:

```jsonc
POST /v1/comfy/provision
{ "workflow": { },   // workflow full-graph; mảng `models` top-level [{name,url,directory,hash}] là model cần tải
  "wait": false,     // true = chờ cài xong mới trả; false = trả ngay, poll status
  "timeout": 1800 }  // tùy chọn, ghi đè COMFY_PROVISION_TIMEOUT
GET /v1/comfy/provision/status   # {done_count, total_count, in_progress_count, is_processing}
```

- Node cần cài lấy từ `class_type` (API-format) hoặc `nodes[].type` (full-graph) không có trong `/object_info`; ánh xạ sang pack cài được qua `getmappings`. Model lấy từ mảng `models` top-level của workflow full-graph — node còn thiếu nhưng resolve được thì báo trong `unresolved_nodes`.
- `auto_provision: true` khi generate = chạy `provision(wait=true)` trước rồi mới render (chỉ có tác dụng khi gửi kèm `workflow`).
- `ephemeral: true` = đổi `SaveImage`→`PreviewImage` (ảnh vào thư mục temp, không vào gallery cố định) **và** xóa entry history sau khi lấy ảnh xong; ảnh vẫn trả về (url/b64/binary) như thường. Bật global bằng `COMFY_EPHEMERAL=true`.

**Nhập ảnh/mask vào ComfyUI (img2img / ControlNet / inpaint)** — phân tích workflow để biết node nào cần file, rồi upload:

```jsonc
POST /v1/comfy/analyze
{ "workflow": { } }   // -> { "slots": [{node_id, class_type, input_name, upload_kind, current_value}] }
// mỗi slot là 1 input nhận file (LoadImage.image…), dò bằng cờ *_upload trong /object_info

POST /v1/comfy/upload
{ "image": "data:image/png;base64,…" | "https://…",  // ảnh cần nhập
  "filename": "ref.png",   // tùy chọn; mặc định myrouter_<hex>.<ext>
  "ephemeral": true,       // upload vào thư mục temp (mặc định COMFY_EPHEMERAL)
  "mask": false,           // true = /upload/mask, ghép vào alpha của original_ref
  "original_ref": { } }    // {filename, subfolder, type} của ảnh đã upload (khi mask=true)
// -> { name, subfolder, type, ref }   // `ref` cắm thẳng vào input LoadImage
```

- Luồng: `analyze` → với mỗi slot `upload` một file → lấy `ref` gán vào `workflow[node_id].inputs[input_name]` → `generate`.
- `ref` = tên file (bare) khi vào `input/`, hoặc `"tên [temp]"` khi ephemeral (ComfyUI hiểu annotation `[temp]`/`[output]`).
- **Ephemeral upload = thư mục temp** (tự dọn, không vào `input/` cố định). ComfyUI **không có API xóa file input** ngay lập tức, nên temp là cơ chế "không lưu" khả dụng (giống ephemeral output).

**Microsoft Copilot** (copilot key; nhiều tài khoản MS — mỗi key gắn 1 account):

```jsonc
POST /copilot/v1/chat/completions
{ "model": "copilot",
  "stream": false,                      // true = SSE token-by-token
  "conversation_id": "new",             // "new" tạo mới; id cũ để chat tiếp; bỏ trống = stateless
  "messages": [
    { "role": "user", "content": [      // 1 ảnh input (Copilot đọc được)
        { "type": "text", "text": "Ảnh này là gì?" },
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,…" } } ] } ],
  "tools": [ /* … */ ]                   // function calling: GIẢ LẬP như Gemini (TOOL_EMULATION)
}
GET /copilot/v1/models                   # luôn trả 1 model "copilot"
GET /copilot/v1/conversations            # lịch sử thread (sidecar tự lưu)
DELETE /copilot/v1/conversations/{id}
```

- **Năng lực** (đúng những gì thư viện hỗ trợ): text, **streaming**, **conversation**, **ảnh output** (`message.images`), **function calling giả lập** (`app/tools.py`, như Gemini). Không có: chọn model/mode (mode cố định `"smart"`), upload file khác ảnh, web-search/plugin.
- **Chat chạy qua headless browser** (`COPILOT_CHAT_MODE=browser`, mặc định): pure-HTTP (curl_cffi) bị Cloudflare chặn ở một số môi trường — JA3 của curl_cffi không tái dùng được `cf_clearance` mà Chromium mới hơn earn, nên mọi lượt 503. Browser (headless) thì **qua được Cloudflare** (đã verify), nên MyRouter gõ prompt vào composer và đọc reply từ frame `appendText` của chat WebSocket. Chậm hơn HTTP + serialize mỗi account, nhưng chạy được ở nơi HTTP bị chặn. Đặt `COPILOT_CHAT_MODE=http` để quay lại driver curl_cffi (nhanh hơn nếu môi trường cho phép). Ảnh input chưa hỗ trợ ở chế độ browser.
- **Đăng nhập**: dashboard **Status → Copilot Profiles → Add account & login** → cửa sổ browser Microsoft/Google mở trên máy chủ (**cần màn hình**). Session lưu ở `COPILOT_SESSION_ROOT/<name>/` (git-ignored), DB chỉ giữ trạng thái. **Với tài khoản Google** (federated): sau khi đăng nhập, **gửi 1 tin nhắn** trong cửa sổ — token chat của Google chỉ được cấp ở lượt chat đầu tiên (MSAL cache bị mã hóa), nên đây là bước bắt buộc để hoàn tất (cửa sổ tự đóng, profile → active). Tài khoản Microsoft thì tự động xong.
- **Cloudflare clearance (~30 phút)**: chỉ lấy được bằng **browser thật**. Trên máy **có màn hình** (PM2 desktop): đặt `COPILOT_INTERACTIVE_CLEAR=true` → khi clearance hết hạn, MyRouter tự mở browser làm mới rồi retry request (chờ ~30s), không trả 503. Trên **VPS headless**: để `false` (mặc định) → clearance hết hạn trả `503 clearance_required`, phải login lại trên máy có màn hình (chia sẻ session dir). `COPILOT_HEADLESS_CLEAR=true` thử làm mới headless im lặng trước (không ổn định trên IP datacenter/VPN).
- **Ephemeral mặc định** (`CHAT_TEMPORARY=true`): chat stateless (không kèm `conversation_id`) chạy ở chế độ **temporary — không lưu vào lịch sử web**. Gemini hỗ trợ native; Copilot best-effort (xóa conversation upstream sau khi trả lời — backend không có cờ temporary). Gửi kèm `conversation_id` → luôn được lưu (thread cần persist để tiếp tục). Temporary do sự có mặt của `conversation_id` quyết định — không có field override riêng từng request.
- **Serialize**: một account xử lý tuần tự (~1–4 request đồng thời); MyRouter khóa `asyncio.Lock` mỗi account. Đây là cầu nối cá nhân, không phải gateway throughput cao.

### Năng lực chat (Gemini)

- **Text / streaming / conversation**: đầy đủ (xem "Streaming vs non-streaming").
- **Ephemeral mặc định** (`CHAT_TEMPORARY=true`): chat stateless dùng `generate_content(temporary=True)` → không lưu vào lịch sử Gemini web. Gửi kèm `conversation_id` → lưu bình thường và tiếp tục được. Playground có checkbox "Temporary" (bật = không gắn `conversation_id`).
- **Vision** (ảnh input `image_url`): **chạy được** — sidecar tách part ảnh, ghi file tạm đúng MIME, truyền vào Gemini (`VISION_MAX_IMAGE_MB` giới hạn kích thước). Khi account bị Google rate-limit tạm thời có thể gặp `APIError 1096/1100` (thử lại sau).
- **Ảnh trong response** (Gemini trả về) → `message.images` (`[{url, title, alt, kind}]`), playground render `<img>`:
  - **web images** (URL công khai): truyền thẳng URL.
  - **generated images** (Gemini tự tạo): sidecar tải qua session tối giản (chỉ `__Secure-1PSID`/`PSIDTS` — dùng full jar sẽ bị CDN 403) rồi **nhúng base64** để browser render. Ảnh nào hiếm khi vẫn không tải được thì hiện placeholder.
  - Field `images` ngoài chuẩn OpenAI nên SDK thường bỏ qua an toàn.
- **Function calling** (`tools` → `tool_calls`): **giả lập bằng prompt** — backend Gemini web không có tool API native. Sidecar nhét schema `tools` vào prompt, model xuất JSON, parse ngược thành `tool_calls` chuẩn OpenAI (`finish_reason:"tool_calls"`); hỗ trợ `tool_choice` auto/required/none/chỉ-định và round-trip `role:"tool"`. Chạy tốt với gemini-3-pro nhưng best-effort (phụ thuộc model bám format). Tắt bằng `TOOL_EMULATION=false`.

### Tích hợp 9Router — 3 provider surface riêng

| Provider | Base URL | Auth | `GET /models` trả về |
|---|---|---|---|
| Gemini | `http://<host>:<port>/gemini/v1` | google key qua header | `gemini`, `gemini-3-*` |
| NotebookLM | `http://<host>:<port>/notebooklm/<api-key>/v1` | **key nằm trong URL** — không cần header | toàn bộ notebook id + tiêu đề của profile |
| ComfyUI (mỗi instance) | `http://<host>:<port>/comfyui/v1` | comfy key qua header | đúng instance của key |
| Copilot (mỗi account) | `http://<host>:<port>/copilot/v1` | copilot key qua header | `copilot` |

- Mỗi surface chỉ phục vụ backend của nó: gửi notebook id vào `/gemini/v1` (hoặc `gemini` vào `/notebooklm/...`) → 404 kèm chỉ dẫn sang surface đúng.
- `/gemini/v1` có cả `/conversations`; `/comfyui/v1` có cả `/comfy/info` + `/comfy/queue` + `/comfy/provision` (+ `/status`) + `/comfy/analyze` + `/comfy/upload`; `/copilot/v1` có cả `/conversations`; lệnh artifact NotebookLM vẫn ở surface gộp (`/v1/notebooklm/*`).
- Bề mặt gộp cũ `/v1/*` giữ nguyên (playground, OpenAI SDK, provider cũ).
- 9Router chạy trong Docker: dùng `http://172.17.0.1:<port>` (bridge IP) thay vì `localhost`.
- Lưu ý: key trên URL của surface NotebookLM sẽ xuất hiện trong access log — chỉ dùng trong mạng tin cậy.
- Output Format "JSON (Base64)"/"Binary File" và Size "auto" của form ảnh 9Router được hỗ trợ sẵn.

### Streaming vs non-streaming

Sidecar hỗ trợ **cả hai** trên mọi surface, theo chuẩn OpenAI — quyết định bằng field `stream` trong body:

| `stream` | Sidecar trả về |
|---|---|
| `true` | SSE: chunk `role` (ra ngay, TTFT thấp) → các chunk `content` delta tăng dần (Gemini stream thật) → chunk `finish` mang `usage` → `data: [DONE]`. NotebookLM: một chunk trả lời + finish/usage. |
| bỏ trống / `false` | Một object JSON `chat.completion` sạch (không SSE, không `[DONE]`). |

Gọi **thẳng sidecar** thì cả hai chế độ đều sạch từng byte:

```js
// STREAMING — đọc SSE, bỏ qua [DONE]
const res = await fetch(`${BASE}/v1/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "Authorization": `Bearer ${key}` },
  body: JSON.stringify({ model: "gemini-3-pro", stream: true,
                         messages: [{ role: "user", content: "Xin chào" }] })
});
const reader = res.body.getReader(); const dec = new TextDecoder();
let full = "", buf = "";
while (true) {
  const { done, value } = await reader.read(); if (done) break;
  buf += dec.decode(value, { stream: true });
  const lines = buf.split("\n"); buf = lines.pop();
  for (const l of lines) {
    const t = l.trim(); if (!t.startsWith("data:")) continue;
    const p = t.slice(5).trim(); if (p === "[DONE]") continue;
    const d = JSON.parse(p).choices?.[0]?.delta?.content; if (d) full += d;
  }
}

// NON-STREAM — JSON thuần
const data = await (await fetch(`${BASE}/v1/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "Authorization": `Bearer ${key}` },
  body: JSON.stringify({ model: "gemini-3-pro",
                         messages: [{ role: "user", content: "Xin chào" }] })
})).json();   // data.choices[0].message.content
```

**Qua 9Router có một lưu ý:** 9Router luôn stream tới upstream. Client **streaming** (Cursor, Claude Code, Cline, Copilot — chế độ 9Router sinh ra để phục vụ) chạy tốt vì 9Router forward thẳng SSE. Nhưng khi client gọi **non-stream** (`response.json()`), 9Router gom SSE lại thành JSON rồi **ghép nhầm `data: [DONE]` vào sau** → `response.json()` lỗi *"Unexpected non-whitespace character after JSON"*. Đây là **bug của 9Router** (content + usage đã gom đúng, chỉ dư terminator; đã xác nhận 9Router tự hardcode terminator này — sidecar không gỡ được). Cách xử lý:
- Ưu tiên **dùng streaming** qua 9Router (đúng thiết kế), hoặc
- **Trỏ client thẳng vào sidecar** cho các provider này (mất fallback/RTK của 9Router nhưng cả hai chế độ sạch), hoặc
- Parse phòng thủ ở client: `JSON.parse(text.split('data: [DONE]')[0].trim())`.

### Verify nó chạy

Script kiểm tra cả hai chế độ (in ra nội dung thật), cần Node 18+, không cần cài gì:

```bash
# Qua 9Router (non-stream sẽ tự cắt đuôi "data: [DONE]")
node scripts/verify.mjs https://9router.montserrat.id.vn/v1 <mr-key> mr/gemini-3-flash

# Hoặc trỏ thẳng sidecar (sạch cả hai chiều)
node scripts/verify.mjs http://localhost:8000/v1 <google-key> gemini-3-flash
```

Kỳ vọng: cả **NON-STREAM** lẫn **STREAMING** đều in ra content + `usage`. Nếu cái test SSE của bạn trước đó hiện "(empty)" thì đó là do công cụ chỉ theo dõi kết nối chứ không parse `data:` — script này parse đúng nên sẽ hiện nội dung.

Bản browser-console tương đương (dán vào Console, sửa `URL`/`KEY`):

```js
const URL_ = "https://9router.montserrat.id.vn/v1/chat/completions", KEY = "<mr-key>";
const H = { "Content-Type": "application/json", Authorization: `Bearer ${KEY}` };
const body = m => JSON.stringify({ model: "mr/gemini-3-flash", ...m, messages: [{ role: "user", content: "Reply with exactly: it works" }] });
// non-stream (cắt đuôi rác)
const t = await (await fetch(URL_, { method: "POST", headers: H, body: body({}) })).text();
console.log("NON-STREAM:", JSON.parse(t.split("data: [DONE]")[0].trim()).choices[0].message.content);
// streaming
const r = await fetch(URL_, { method: "POST", headers: H, body: body({ stream: true }) });
const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "", out = "";
while (true) { const { done, value } = await rd.read(); if (done) break;
  buf += dec.decode(value, { stream: true }); const ls = buf.split("\n"); buf = ls.pop();
  for (const l of ls) { const s = l.trim(); if (!s.startsWith("data:")) continue;
    const p = s.slice(5).trim(); if (p === "[DONE]") continue;
    try { const d = JSON.parse(p).choices?.[0]?.delta?.content; if (d) out += d; } catch {} } }
console.log("STREAMING:", out);
```

---

## 5. Vận hành & xử lý sự cố

| Hiện tượng | Nguyên nhân / Cách xử lý |
|---|---|
| 502 `profile_auth_expired` | Auth hết hạn và auto re-login chưa xong — nếu có cửa sổ browser đang mở trên máy chủ, đăng nhập nốt rồi gọi lại; hoặc vào Status bấm Re-login. Cooldown 2 phút giữa các lần tự thử. |
| Cửa sổ Edge tự bật khi gọi API | Là auto re-login (thường tự đóng ~5s). Tắt bằng `AUTO_RELOGIN=false`. |
| 403 `wrong_key_type` | Dùng nhầm loại key (google ↔ comfy ↔ copilot). |
| Copilot `503 clearance_required` | Cloudflare clearance hết hạn, không tự làm mới headless được — vào Status → Copilot Profiles → Re-login trên máy **có màn hình** (chia sẻ session dir). Xem mục Copilot. |
| Copilot `502 profile_auth_expired` / `503 profile_not_authenticated` | Account chưa login hoặc session hết hạn — login lại ở Status. |
| Chat Gemini không hiện trong lịch sử web | Đã fix (cache cookie degraded); nếu tái diễn: Re-login profile, kiểm tra log có `UNAUTHENTICATED`. |
| `model_not_found` khi gọi Gemini | Xem model hợp lệ qua `GET /v1/models`. |
| 9Router trả JSON dính `data: [DONE]` (non-stream) | Bug aggregate của 9Router — xem mục "Streaming vs non-streaming". Dùng streaming, hoặc trỏ thẳng sidecar, hoặc parse phòng thủ. |
| ComfyUI 502/504 | Tunnel down hoặc checkpoint không tồn tại — kiểm tra `GET /v1/comfy/info`, `/v1/comfy/queue`. |
| Theo dõi | Bảng **Request Logs** trên dashboard (endpoint, status, latency, error); trang **Status** có thống kê 24h + reachability. |

**Bảo mật**: DB chứa API key và cookie Google (bảng `google_profiles.storage_state`); session Copilot (cookie + MSAL token + browser profile) nằm ở `COPILOT_SESSION_ROOT/` trên đĩa (đã git-ignore). Chỉ chạy trên máy tin cậy, đặt mật khẩu MySQL/admin thật, không expose `/admin` ra ngoài mạng nội bộ.
