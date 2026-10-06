# Vision Service Core — Lower-body & Foot Motion Tracking

ระบบ Computer Vision แบบ Real-time สำหรับตรวจจับการเคลื่อนไหวของร่างกายส่วนล่าง (สะโพก → ปลายเท้า)
แล้วส่งพิกัดและ Gesture Event ออกทาง **UDP `127.0.0.1:5005`** ให้ Game Client (Unity / Godot / Unreal / Python) นำไปใช้

```
Camera ─► CameraStream ─► PoseBackend ─► LandmarkSmoother ─► BodyNormalizer ─► GestureRecognizer ─► MotionPacket ─► UdpSender
 (thread,     (MediaPipe /     (One Euro /       (hip-relative,      (STEP/STOMP/KICK/      (JSON /         (non-blocking,
 latest-frame)  YOLO / hybrid)   EMA)             leg-length units)   JUMP state machine)    binary)         port 5005)
```

## โครงสร้างโปรเจกต์

| ไฟล์ | หน้าที่ |
|---|---|
| `vision_service/capture.py` | เปิดกล้องผ่าน OpenCV (MJPG, 640×480@60, buffer=1) ด้วย thread แยก เก็บเฉพาะเฟรมล่าสุด เพื่อตัดดีเลย์จาก buffer |
| `vision_service/landmarks.py` | นิยามจุดข้อต่อ 10 จุด (Hip, Knee, Ankle, Heel, Foot Index × ซ้าย/ขวา) และ mapping ไป MediaPipe / COCO |
| `vision_service/pose/mediapipe_backend.py` | MediaPipe Pose (BlazePose) ผ่าน Tasks API โหมด VIDEO (ดาวน์โหลดโมเดลอัตโนมัติ) รองรับ `mp.solutions` รุ่นเก่า |
| `vision_service/pose/yolo_backend.py` | YOLOv8-Pose สำรอง (COCO-17 ไม่มีส้น/ปลายเท้า → ประมาณจากทิศหน้าแข้งและติดธง `estimated`) |
| `vision_service/pose/hybrid.py` | สลับ MediaPipe ↔ YOLO อัตโนมัติเมื่อหาคนไม่เจอหรือเท้าเคลื่อนที่เร็วเกิน |
| `vision_service/filters.py` | One Euro Filter / EMA / none แบบ vectorized แยกฟิลเตอร์ต่อข้อต่อ พร้อมหยุดค่าเมื่อจุดถูกบัง |
| `vision_service/normalization.py` | แปลงเป็นพิกัดสัมพัทธ์กับสะโพก หน่วย "ความยาวขา" (ทุกส่วนสูงได้ค่าเท่ากัน) + มุมเข่า |
| `vision_service/gestures.py` | ตรรกะ Step / Stomp / Kick / Jump |
| `vision_service/protocol.py` | สัญญาข้อมูล (JSON + Binary) และตัว decode อ้างอิง |
| `vision_service/transport.py` | UDP sender แบบ non-blocking |
| `vision_service/debug_overlay.py` | หน้าจอ Debug: โครงกระดูกท่อนล่าง, เส้นพื้น, แถบความสูงเท้า, **คำเตือนเมื่อไม่พบสะโพก** |
| `vision_service/service.py`, `__main__.py` | main loop + CLI |
| `tools/udp_monitor.py` | ตัวรับ UDP อ้างอิง (ใช้ทดสอบแทนเกม) วัด packet loss และ latency |
| `tests/` | unit test 31 รายการ (ไม่ต้องใช้กล้อง) |

## ติดตั้งและรัน

```bash
cd foot_motion_vision
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
# (ทางเลือก) โมเดลสำรองความเร็วสูง:  pip install ultralytics

python -m vision_service                     # กล้อง 0, หน้าต่าง debug, JSON → 127.0.0.1:5005
python tools/udp_monitor.py                  # อีก terminal: ดูข้อมูลที่เกมจะได้รับ
python -m pytest -q tests                    # รันชุดทดสอบ
```

> Linux แบบ headless ต้องมี `libegl1` `libgles2` สำหรับ MediaPipe Tasks

ตัวเลือกที่ใช้บ่อย:

```bash
python -m vision_service --backend hybrid --device cuda:0     # MediaPipe + YOLOv8 fallback
python -m vision_service --variant lite                        # CPU ช้า: ใช้โมเดล lite
python -m vision_service --format binary --no-debug            # production: binary, ไม่มีหน้าต่าง
python -m vision_service --camera clip.mp4                     # ทดสอบจากไฟล์วิดีโอ
python -m vision_service --filter ema --ema-alpha 0.6
python -m vision_service --config config.example.json          # ปรับ threshold ทั้งหมดจากไฟล์
python -m vision_service --exposure -7                         # ลด motion blur (ค่าตาม driver)
```

ปุ่มลัดในหน้าต่าง Debug: `q`/`Esc` ออก · `f` สลับฟิลเตอร์ (one_euro → ema → none) · `r` รีเซ็ต (เรียนรู้พื้น/สเกลใหม่) · `m` สลับภาพกระจก

## การตั้งกล้อง

- สูงจากพื้น **30–50 ซม.** (ระดับเข่า/หน้าแข้ง) เงยขึ้น **5–10°**
- ต้องเห็นตั้งแต่ **สะโพก/เอว ลงไปถึงพื้น** — สะโพกคือจุดอ้างอิงของพิกัดทั้งหมด หากไม่เห็นจะขึ้นแถบแดง `HIP NOT DETECTED` และ packet จะมี `"tracking": false`
- ยืนห่างกล้องประมาณ 1.5–2.5 ม. ให้เท้าทั้งสองอยู่ในเฟรมขณะเตะ
- แสงสว่างพอและตั้ง exposure สั้น จะลดภาพเบลอตอนกระทืบเท้า
- ช่วงเริ่มต้น ยืนนิ่ง ~0.5 วินาที เพื่อให้ระบบเรียนรู้ระดับพื้นและความยาวขา

## ระบบพิกัด (ส่งออก)

| แกน | ความหมาย |
|---|---|
| origin | จุดกึ่งกลางสะโพกซ้าย-ขวา |
| +x | ด้าน **ขวาของผู้เล่น** (เท้าซ้ายจึงเป็นค่าลบ) |
| +y | **ลง** (ข้อเท้าตอนยืน ≈ +1.0) |
| +z | ห่างออกจากกล้อง (MediaPipe เท่านั้น; YOLO = 0) |
| หน่วย | 1.0 = ความยาวขาของผู้เล่น (สะโพก→เข่า + เข่า→ข้อเท้า) |

`velocity_y` = ความเร็วแนวดิ่งของเท้า หน่วยความยาวขา/วินาที, **บวก = ลง** (ยกเท้าขึ้น → ค่าลบ)

## Actions

| Action | เงื่อนไข |
|---|---|
| `IDLE` | ไม่มี event ใหม่ในเฟรมนี้ |
| `STEP_LEFT` / `STEP_RIGHT` | เท้ายกสูงกว่าพื้น > 0.05 (และสูงกว่าเท้าอีกข้าง) → พุ่งลง → **หยุดชะงัก** ที่พื้น (หรือแตะพื้นครบ 100 ms) |
| `STOMP_LEFT` / `STOMP_RIGHT` | เหมือน Step แต่ความเร็วพุ่งลงสูงสุด ≥ 2.0 ขา/วินาที |
| `KICK_LEFT` / `KICK_RIGHT` | ยกเท้า ≥ 0.30 + ดีดขึ้นเร็ว ≥ 1.6 + มุมเข่า ≥ 140° (ขาเหยียด), หรือเตะข้างออกไป ≥ 0.55 จากแนวสะโพก. การวางเท้าหลังเตะ **ไม่** นับเป็น Step |
| `JUMP` | เท้าทั้งสองลอยจากพื้น + สะโพกยกขึ้น ≥ 0.08 ด้วยความเร็ว ≥ 0.8. การลงพื้นหลังกระโดดไม่นับเป็น Step |

การตรวจ Step/Jump ใช้ "พื้นที่เรียนรู้ได้" ในพิกัดภาพ (กล้องอยู่นิ่ง) จึงไม่หลอกด้วยการย่อตัว.
การตรวจ Kick ใช้พิกัดสัมพัทธ์สะโพก + มุมเข่า 3D (z จาก MediaPipe ช่วยแยก "ยกเข่าสูง" ออกจาก "เตะ").
ทุก threshold อยู่ใน `GestureConfig` และ override ได้จาก `--config`

## Data Streaming Contract (UDP :5005)

### JSON (ค่าเริ่มต้น, ~350 bytes)

```json
{
  "v": 1,
  "seq": 1234,
  "timestamp": 1728144000.123,
  "action": "STEP_LEFT",
  "tracking": true,
  "backend": "mediapipe",
  "latency_ms": 11.8,
  "feet_pos": {
    "left":  {"x": -0.15, "y": 0.85, "z": 0.02},
    "right": {"x": 0.20,  "y": 0.90, "z": -0.01}
  },
  "velocity_y": {"left": -1.45, "right": 0.02},
  "events": [{"id": 57, "type": "STEP_LEFT", "foot": "left", "intensity": 1.3, "x": -0.15}]
}
```

- `timestamp` — เวลา (Unix, วินาที) ที่กล้องจับภาพ
- `latency_ms` — เวลาตั้งแต่จับภาพจนส่ง packet (ภายใน vision service)
- `action` — event สำคัญที่สุดที่ **เกิดในเฟรมนี้** (ลำดับ JUMP > KICK > STOMP > STEP) หรือ `IDLE`
- `events` — event แต่ละตัวถูก **ส่งซ้ำ 3 packet ติดกัน** กัน UDP หาย → ฝั่งเกมต้อง de-duplicate ด้วย `id`
- `seq` — เพิ่มขึ้นทีละ 1; ฝั่งเกมควรทิ้ง packet ที่ `seq` ≤ ค่าล่าสุด
- `--include-joints` เพิ่ม field `joints` (ข้อต่อทั้ง 10 จุด) สำหรับ debug / avatar

### Binary (`--format binary`, header 56 bytes + 14 bytes/event, little-endian)

| offset | type | field |
|---|---|---|
| 0 | `char[4]` | magic `"LBMT"` |
| 4 | `u8` | version (=1) |
| 5 | `u8` | flags: bit0 tracking, bit1 YOLO backend |
| 6 | `u8` | action code |
| 7 | `u8` | event_count |
| 8 | `u32` | seq |
| 12 | `f64` | timestamp |
| 20 | `f32×3` | left x, y, z |
| 32 | `f32×3` | right x, y, z |
| 44 | `f32×2` | velocity_y left, right |
| 52 | `f32` | latency_ms |
| 56 | — | event × N: `u32 id, u8 action, u8 foot (0 none/1 L/2 R), f32 intensity, f32 x` |


Action codes: `0 IDLE, 1 STEP_LEFT, 2 STEP_RIGHT, 3 STOMP_LEFT, 4 STOMP_RIGHT, 5 KICK_LEFT, 6 KICK_RIGHT, 7 JUMP`

### ตัวอย่างฝั่งเกม (Unity C#) — สำหรับเฟสถัดไป

```csharp
using System.Net; using System.Net.Sockets; using UnityEngine;

public class MotionReceiver : MonoBehaviour {
    UdpClient udp; uint lastSeq; uint lastEventId;
    void Start() { udp = new UdpClient(5005); udp.Client.Blocking = false; }
    void Update() {
        var ep = new IPEndPoint(IPAddress.Any, 0);
        while (udp.Available > 0) {                       // drain: keep only newest state
            byte[] d = udp.Receive(ref ep);
            if (d.Length < 56 || d[0] != 'L') continue;
            uint seq = System.BitConverter.ToUInt32(d, 8);
            if (seq <= lastSeq) continue; lastSeq = seq;
            bool tracking = (d[5] & 1) != 0;
            var left  = new Vector3(System.BitConverter.ToSingle(d, 20), -System.BitConverter.ToSingle(d, 24), System.BitConverter.ToSingle(d, 28));
            var right = new Vector3(System.BitConverter.ToSingle(d, 32), -System.BitConverter.ToSingle(d, 36), System.BitConverter.ToSingle(d, 40));
            int n = d[7];
            for (int i = 0; i < n; i++) {
                int o = 56 + i * 14;
                uint id = System.BitConverter.ToUInt32(d, o);
                if (id <= lastEventId) continue; lastEventId = id;   // de-dup repeated events
                byte action = d[o + 4];
                // TODO: dispatch action to gameplay
            }
        }
    }
    void OnDestroy() => udp?.Close();
}
```

## การปรับจูน (Tuning)

**One Euro Filter** (ค่าเริ่มต้นจูนสำหรับเท้า 640×480@60 FPS ในหน่วย pixel):

| พารามิเตอร์ | ค่า | ผล |
|---|---|---|
| `min_cutoff` | 1.0 Hz | ↓ = นิ่งขึ้นตอนเท้าอยู่กับที่ แต่หน่วงขึ้นตอนขยับช้า |
| `beta` | 0.05 | ↑ = หน่วงน้อยลงตอนขยับเร็ว (กระทืบ/เตะ) แต่สั่นขึ้น |
| `d_cutoff` | 2.0 Hz | ความเร็วตอบสนองของตัวประมาณความเร็ว |

จากการทดสอบ: jitter ของเท้าที่นิ่งลดลง ~8 เท่า ขณะที่ความเร็วสูงสุดของการกระทืบ 0.05 วินาทียังคงอยู่ >95%.
ถ้าใช้ 1280×720 ให้ลด `beta` ลงประมาณครึ่งหนึ่ง (beta คูณกับ px/s).
ฟิลเตอร์ทั้งสองเป็น IIR อันดับหนึ่ง (state เดียวต่อค่า) จึงไม่มีดีเลย์สะสม

**ขั้นตอนจูน:** ยืนนิ่ง → ลด `min_cutoff` จนจุดไม่สั่น → กระทืบเร็ว ๆ → เพิ่ม `beta` จนจุดตามเท้าทัน → ปรับ `stomp_speed` / `kick_*` ตามผู้เล่นจริง (ดูค่า vy ใต้แถบเท้าในหน้าจอ debug)

## Latency

| ช่วง | ค่าที่วัดได้ |
|---|---|
| UDP localhost (send → recv + decode) | median < 0.2 ms (test บังคับ < 2 ms) |
| `sendto` แบบ non-blocking | ไม่กี่ไมโครวินาที, ไม่บล็อกแม้ไม่มีตัวรับ |
| Pose inference (MediaPipe full, CPU) | ~10–30 ms ขึ้นกับเครื่อง (`--variant lite` เร็วกว่า) |

ช่วงที่ใช้เวลามากที่สุดคือ pose inference ไม่ใช่การส่งข้อมูล — `latency_ms` ใน packet และ `tools/udp_monitor.py` แสดงตัวเลขจริงบนเครื่องของคุณ

## ข้อจำกัดที่ทราบ

- กล้องหน้าตรงมุมต่ำ: การยกเข่าสูงกับการเตะตรงไปหากล้องดูคล้ายกันในภาพ 2D — ระบบใช้ z ของ MediaPipe ช่วยแยก; ในโหมด YOLO (ไม่มี z) อาจแยกได้น้อยลง
- YOLO ไม่มีส้นเท้า/ปลายเท้า จุดเหล่านี้ถูกประมาณ (แสดงเป็นสีเทาในหน้าจอ debug)
- รองรับผู้เล่น 1 คน (เลือกคนที่ใหญ่ที่สุดในเฟรม)
