"""Live MJPEG view + continuous timestamped H.264 recording of a USB webcam.

One capture thread grabs frames; the recorder burns an Eastern wall-clock stamp into
each frame and pipes it to ffmpeg (NVENC, fallback libx264) in ~10-minute fragmented-MP4
segments, with a per-frame CSV sidecar. The HTTP server shows the live view.

Video time tracks wall time: frames are duplicated/dropped onto a fixed 30 fps grid, so
frame k of a segment is at segment_start + k/30 s. The sidecar gives, for every written
frame, the capture time of the camera frame used.

Machine-specific defaults (ffmpeg, the recordings folder, the folders a run may record
into) are agni's; override them with the environment variables named below or the flags.
"""

# ruff: noqa: PLR0917 -- a copy of the recorder service that runs on agni, kept close to
# it; its helpers take positional arguments.

import argparse
import datetime as dt
import fcntl
import json
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from zoneinfo import ZoneInfo

import cv2

ET = ZoneInfo("America/New_York")
FFMPEG = os.environ.get("WEBCAM_FFMPEG", "/home/josh/miniconda3/envs/ffnv/bin/ffmpeg")
RECORDINGS = os.environ.get(
    "WEBCAM_RECORDINGS",
    "/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/webcam/recordings",
)
# A run (`/record/start`) may record only into a folder under one of these.
RUN_ROOTS = tuple(
    os.environ.get("WEBCAM_RUN_ROOTS", os.path.expanduser("~") + "/:/tmp/").split(":")
)


def et_now(t):
    return dt.datetime.fromtimestamp(t, ET)


class Camera:
    def __init__(self, dev, w, h, fps):
        self.dev, self.w, self.h, self.fps = dev, w, h, fps
        self.cond = threading.Condition()
        self.frame = None
        self.t = 0.0
        self.seq = 0
        self.fps_meas = 0.0
        # Every frame also goes to the recorder in order, so an encoder stall
        # (NVENC init at each segment start) is absorbed instead of dropping frames.
        self.rec_q = queue.Queue(maxsize=90)
        self.rec_drops = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        c = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        c.set(cv2.CAP_PROP_FPS, self.fps)
        self._no_exposure_priority()
        return c

    def _no_exposure_priority(self):
        # C920 default exposure_auto_priority=1 lets auto-exposure drop the frame
        # rate in dim light; 0 keeps auto-exposure but holds the requested fps.
        try:
            fd = os.open(self.dev, os.O_RDWR)
            try:
                fcntl.ioctl(fd, 0xC008561C, struct.pack("Ii", 0x009A0903, 0))  # S_CTRL
            finally:
                os.close(fd)
        except OSError as e:
            print("could not clear exposure_auto_priority:", e, flush=True)

    def _run(self):
        cap = self._open()
        win_t, win_n = time.time(), 0
        fails = 0
        while True:
            ok, f = cap.read()
            t = time.time()
            if not ok:
                fails += 1
                if fails > 30:
                    print("camera read failing; reopening", flush=True)
                    cap.release()
                    time.sleep(1)
                    cap = self._open()
                    fails = 0
                continue
            fails = 0
            with self.cond:
                self.frame, self.t = f, t
                self.seq += 1
                self.cond.notify_all()
                seq = self.seq
            try:
                self.rec_q.put_nowait((f, t, seq))
            except queue.Full:
                self.rec_drops += 1
            win_n += 1
            if t - win_t >= 2.0:
                self.fps_meas = win_n / (t - win_t)
                win_t, win_n = t, 0

    def wait(self, last_seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout=timeout)
            return self.frame, self.t, self.seq


def stamp(img, t):
    s = et_now(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " ET"
    h = img.shape[0]
    scale = h / 1080 * 1.0
    th = max(1, int(round(2 * scale)))
    (tw, tht), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = 12, h - 14
    cv2.rectangle(img, (x - 6, y - tht - 8), (x + tw + 6, y + base + 4), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img


class Recorder:
    def __init__(self, cam, root, seg_s, out_fps, codec_pref):
        self.cam, self.root, self.seg_s, self.out_fps = cam, root, seg_s, out_fps
        self.codec = codec_pref
        self.lock = threading.Lock()
        self.want = True
        self.cur_file = None
        self.seg_start = None
        self.err = None
        # An experiment run (launch.sh) forces recording on and gets its own segments,
        # cut at its start and end, written to <run dir>/webcam/ and symlinked into the
        # continuous archive. The run ends on /record/stop or when its pid exits.
        self.run = None
        self.rotate = False
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._watch_run, daemon=True).start()

    def status(self):
        with self.lock:
            el = time.time() - self.seg_start if self.seg_start else 0
            run = dict(self.run) if self.run else None
            if run:
                run["elapsed_s"] = time.time() - run["start_unix_s"]
            return {
                "recording": self.cur_file is not None,
                "want": self.want,
                "file": self.cur_file,
                "elapsed_s": el,
                "codec": self.codec,
                "error": self.err,
                "run": run,
            }

    def set(self, on):
        with self.lock:
            if not on and self.run:
                return False
            self.want = on
            return True

    def start_run(self, run_dir, pid):
        os.makedirs(os.path.join(run_dir, "webcam"), exist_ok=True)
        with self.lock:
            self.run = {
                "dir": run_dir,
                "pid": pid,
                "start_unix_s": time.time(),
                "start_et": et_now(time.time()).isoformat(timespec="seconds"),
            }
            self.rotate = True
        print(f"run start dir={run_dir} pid={pid}", flush=True)

    def stop_run(self, run_dir=None, why="stop"):
        with self.lock:
            if not self.run or (run_dir and run_dir != self.run["dir"]):
                return False
            print(f"run end ({why}) dir={self.run['dir']}", flush=True)
            self.run = None
            self.rotate = True
            return True

    def _watch_run(self):
        while True:
            time.sleep(2)
            with self.lock:
                run = self.run
            if run and run["pid"] and not os.path.exists(f"/proc/{run['pid']}"):
                self.stop_run(run["dir"], why=f"pid {run['pid']} exited")

    def _ffmpeg(self, path, w, h):
        enc = (
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "26",
                "-b:v",
                "6M",
                "-maxrate",
                "10M",
                "-bufsize",
                "20M",
            ]
            if self.codec == "h264_nvenc"
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "4"]
        )
        cmd = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-framerate",
            str(self.out_fps),
            "-i",
            "-",
            *enc,
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(2 * self.out_fps),
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            path,
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _open_segment(self, t0, w, h):
        d = et_now(t0)
        day = os.path.join(self.root, d.strftime("%Y-%m-%d"))
        os.makedirs(day, exist_ok=True)
        name = "webcam_" + d.strftime("%Y-%m-%d_%H-%M-%S")
        with self.lock:
            run = self.run
        if run:
            out = os.path.join(run["dir"], "webcam")
            os.makedirs(out, exist_ok=True)
            base = os.path.join(out, name)
            for ext in (".mp4", ".csv", ".json"):
                link = os.path.join(day, name + ext)
                if not os.path.lexists(link):
                    os.symlink(base + ext, link)
        else:
            base = os.path.join(day, name)
        proc = self._ffmpeg(base + ".mp4", w, h)
        time.sleep(0.3)
        if proc.poll() is not None and self.codec == "h264_nvenc":
            print("nvenc failed; falling back to libx264", flush=True)
            self.codec = "libx264"
            proc = self._ffmpeg(base + ".mp4", w, h)
        csvf = open(base + ".csv", "w", buffering=1)  # noqa: SIM115 -- open per segment
        csvf.write("frame,video_pts_s,capture_unix_s,capture_et,camera_seq\n")
        meta = {
            "start_unix_s": t0,
            "start_et": d.isoformat(),
            "fps": self.out_fps,
            "width": w,
            "height": h,
            "codec": self.codec,
            "run_dir": run["dir"] if run else None,
            "note": "frame k is at start_unix_s + k/fps; capture_* columns give the "
            "camera frame used (frames duplicated/dropped onto the grid)",
        }
        with open(base + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        with self.lock:
            self.cur_file = os.path.basename(base) + ".mp4"
            self.seg_start = t0
        print(f"segment start {base}.mp4 codec={self.codec}", flush=True)
        return proc, csvf

    def _close(self, proc, csvf):
        try:
            proc.stdin.close()
            proc.wait(timeout=30)
        except Exception as e:  # noqa: BLE001
            print("ffmpeg close error", e, flush=True)
            proc.kill()
        csvf.close()
        with self.lock:
            self.cur_file = None
            self.seg_start = None

    def _run(self):
        proc = csvf = None
        seq = 0
        n = 0
        t0 = 0.0
        while True:
            try:
                frame, t, seq = self.cam.rec_q.get(timeout=2.0)
            except queue.Empty:
                frame = None
                t = time.time()
            with self.lock:
                want = self.want or self.run is not None
                rotate, self.rotate = self.rotate, False
            if proc is not None and (rotate or not want or t - t0 >= self.seg_s):
                self._close(proc, csvf)
                proc = None
            if not want or frame is None:
                continue
            if proc is None:
                h, w = frame.shape[:2]
                t0, n = t, 0
                proc, csvf = self._open_segment(t0, w, h)
            # Frames owed so the grid keeps up with wall time. Tolerate +-1 frame of
            # capture jitter before duplicating or dropping, so a steady 30 fps camera
            # maps 1:1 and drift stays bounded (the CSV has the exact times anyway).
            owed = (t - t0) * self.out_fps + 1 - n
            if owed < 0.5:
                continue
            reps = 1 if owed < 2.5 else int(round(owed))
            reps = min(reps, self.out_fps)  # cap a burst after a stall
            img = stamp(frame.copy(), t)
            data = img.tobytes()
            et_s = et_now(t).isoformat(timespec="milliseconds")
            try:
                for _ in range(reps):
                    proc.stdin.write(data)
                    csvf.write(f"{n},{n / self.out_fps:.4f},{t:.4f},{et_s},{seq}\n")
                    n += 1
                self.err = None
            except (BrokenPipeError, OSError) as e:
                self.err = f"ffmpeg pipe: {e}"
                print(self.err, flush=True)
                self._close(proc, csvf)
                proc = None
                time.sleep(1)


class Live:
    """Encodes the preview JPEG once per camera frame, only while someone watches."""

    def __init__(self, cam, width, quality, max_fps):
        self.cam, self.width, self.q, self.min_dt = cam, width, quality, 1.0 / max_fps
        self.cond = threading.Condition()
        self.jpg, self.seq, self.viewers = None, 0, 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        cs = 0
        last = 0.0
        while True:
            if self.viewers == 0:
                time.sleep(0.1)
                continue
            frame, t, cs = self.cam.wait(cs)
            if frame is None or time.time() - last < self.min_dt:
                continue
            last = time.time()
            h, w = frame.shape[:2]
            small = cv2.resize(
                frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA
            )
            stamp(small, t)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.q])
            if ok:
                with self.cond:
                    self.jpg = buf.tobytes()
                    self.seq += 1
                    self.cond.notify_all()

    def wait(self, last):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last, timeout=2.0)
            return self.jpg, self.seq


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>agni webcam</title><style>
body{margin:0;background:#111;color:#eee;font:15px system-ui,sans-serif}
#bar{display:flex;gap:14px;align-items:center;padding:8px 12px;flex-wrap:wrap}
#dot{width:14px;height:14px;border-radius:50%;background:#555;display:inline-block}
#dot.on{background:#e33;animation:b 1s infinite}@keyframes b{50%{opacity:.35}}
button{font-size:15px;padding:6px 14px;border-radius:6px;border:0;cursor:pointer}
#start{background:#2a7}#stop{background:#c33;color:#fff}
img{display:block;width:100%;max-height:calc(100vh - 52px);object-fit:contain}
.mono{font-family:ui-monospace,monospace}</style></head><body>
<div id=bar><span id=dot></span><span id=state class=mono>...</span>
<button id=start onclick="rec(1)">Start recording</button>
<button id=stop onclick="rec(0)">Stop recording</button>
<span id=fps class=mono></span>
<a href="/snapshot.jpg" target=_blank style="color:#8cf">snapshot</a></div>
<img id=live src="/stream.mjpg" alt="live view">
<script>
function fmt(s){s=Math.floor(s);return String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0')}
async function poll(){try{const r=await fetch('/status');const j=await r.json();
document.getElementById('dot').className=j.recording?'on':'';
document.getElementById('state').textContent=(j.run?('RUN '+j.run.dir.split('/').pop()+' '+fmt(j.run.elapsed_s)+' | '):'')+(j.recording?('REC '+j.file+'  '+fmt(j.elapsed_s)):(j.want?'starting...':'not recording'))+(j.error?' ERR '+j.error:'');
document.getElementById('stop').disabled=!!j.run;
document.getElementById('fps').textContent='camera '+j.camera_fps.toFixed(1)+' fps, '+j.codec;}catch(e){document.getElementById('state').textContent='server unreachable'}}
async function rec(on){await fetch(on?'/start':'/stop',{method:'POST'});poll()}
setInterval(poll,1000);poll();
const im=document.getElementById('live');im.onerror=()=>setTimeout(()=>{im.src='/stream.mjpg?'+Date.now()},1000);
</script></body></html>"""  # noqa: E501 -- the live page's inline script


def make_handler(cam, rec, live, pub):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _status(self):
            s = rec.status()
            s["camera_fps"] = cam.fps_meas
            s["recorder_queue_drops"] = cam.rec_drops
            s["frame_clients"] = pub.clients
            s["viewers"] = live.viewers
            return s

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif p == "/status":
                self._send(200, "application/json", json.dumps(self._status()).encode())
            elif p == "/snapshot.jpg":
                frame, t, _ = cam.wait(-1)
                if frame is None:
                    self._send(503, "text/plain", b"no frame")
                    return
                ok, buf = cv2.imencode(
                    ".jpg", stamp(frame.copy(), t), [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                self._send(200, "image/jpeg", buf.tobytes())
            elif p == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                live.viewers += 1
                seq = 0
                try:
                    while True:
                        jpg, seq = live.wait(seq)
                        if jpg is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n"
                            b"Content-Length: "
                            + str(len(jpg)).encode()
                            + b"\r\n\r\n"
                            + jpg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    live.viewers -= 1
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            p, _, qs = self.path.partition("?")
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n).decode() if n else ""
            q = {k: v[0] for k, v in urllib.parse.parse_qs(qs + "&" + body).items()}
            if p in ("/start", "/stop"):
                ok = rec.set(p == "/start")
                self._send(
                    200 if ok else 409, "application/json", json.dumps(self._status()).encode()
                )
            elif p == "/record/start":
                d = os.path.realpath(q.get("dir", ""))
                if not q.get("dir") or not d.startswith(RUN_ROOTS):
                    why = "dir must be under one of " + ", ".join(RUN_ROOTS)
                    self._send(400, "text/plain", why.encode())
                    return
                rec.start_run(d, int(q.get("pid", 0)))
                self._send(200, "application/json", json.dumps(self._status()).encode())
            elif p == "/record/stop":
                d = os.path.realpath(q["dir"]) if q.get("dir") else None
                ok = rec.stop_run(d)
                self._send(
                    200 if ok else 409, "application/json", json.dumps(self._status()).encode()
                )
            else:
                self._send(404, "text/plain", b"not found")

    return H


# Same protocol and authkey as prpl-tidybot's ceiling-camera publishers
# (marker_detector/publisher.py, third_party/constants.py CONN_AUTHKEY): the client
# sends None, the server answers with the next fresh frame's payload. The key is read
# from `--frame-authkey-file`, never kept in this file.


class FramePublisher:
    """multiprocessing.connection publisher of the latest frame as JPEG.

    Payload: {"jpeg": bytes, "timestamp": wall-clock capture time (unix s),
    "seq": camera frame number, "width", "height"}. Full resolution, no burned-in
    stamp. Each camera frame is encoded at most once, and only when a client asks.
    """

    def __init__(self, cam, host, port, quality, authkey):
        self.cam, self.q = cam, quality
        self.listener = Listener((host, port), authkey=authkey)
        self.lock = threading.Lock()
        self.cache = (None, -1)
        self.clients = 0
        threading.Thread(target=self._accept, daemon=True).start()
        print(f"frame publisher on {host}:{port}", flush=True)

    def _payload(self, last_seq):
        frame, t, seq = self.cam.wait(last_seq)
        if frame is None or seq == last_seq:
            return None, last_seq
        with self.lock:
            if self.cache[1] != seq:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.q])
                if not ok:
                    return None, last_seq
                h, w = frame.shape[:2]
                self.cache = (
                    {"jpeg": buf.tobytes(), "timestamp": t, "seq": seq, "width": w, "height": h},
                    seq,
                )
            return self.cache[0], seq

    def _accept(self):
        while True:
            try:
                conn = self.listener.accept()
            except Exception as e:  # noqa: BLE001  (bad authkey, reset)
                print("publisher accept failed:", e, flush=True)
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        self.clients += 1
        last = 0
        try:
            while True:
                conn.recv()
                payload = None
                while payload is None:
                    payload, last = self._payload(last)
                conn.send(payload)
        except (ConnectionResetError, EOFError, BrokenPipeError, OSError):
            pass
        finally:
            self.clients -= 1
            conn.close()


class NoFramePublisher:
    """Stands in for `FramePublisher` when no authkey file is given."""

    clients = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device", default="/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_6AB51EEF-video-index0"
    )
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--segment-s", type=float, default=600)
    ap.add_argument("--root", default=RECORDINGS)
    ap.add_argument("--codec", default="h264_nvenc", choices=["h264_nvenc", "libx264"])
    ap.add_argument("--live-width", type=int, default=1280)
    ap.add_argument("--live-fps", type=float, default=20)
    ap.add_argument("--frame-port", type=int, default=6004)
    ap.add_argument("--frame-jpeg-quality", type=int, default=80)
    ap.add_argument(
        "--frame-authkey-file",
        default=os.environ.get("WEBCAM_FRAME_AUTHKEY_FILE"),
        help="file holding the frame publisher's authkey; without it, no publisher",
    )
    a = ap.parse_args()
    if not shutil.which(FFMPEG):
        sys.exit(f"missing {FFMPEG}")
    cv2.setNumThreads(2)
    cam = Camera(a.device, a.width, a.height, a.fps)
    rec = Recorder(cam, a.root, a.segment_s, a.fps, a.codec)
    live = Live(cam, a.live_width, 75, a.live_fps)
    if a.frame_authkey_file:
        with open(a.frame_authkey_file, "rb") as f:
            authkey = f.read().strip()
        pub = FramePublisher(cam, "0.0.0.0", a.frame_port, a.frame_jpeg_quality, authkey)
    else:
        pub = NoFramePublisher()
        print("no --frame-authkey-file: frame publisher off", flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(cam, rec, live, pub))
    srv.daemon_threads = True
    print(f"serving on 0.0.0.0:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
