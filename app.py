# app.py — 쇼츠 카피 작업실 (강화판 v1.1)
# ------------------------------------------------------------------------------------------------
# 기능 핵심:
# - CC → Whisper(옵션) → (옵션) OCR
# - DeepL 우선, Argos(설치 시) 폴백, 길이 분할/배치 번역
# - 출력: 평문 TXT / Vrew TXT / 타임코드 TXT(한 줄형) / SRT
# - 파일명: 비디오ID가 아닌 **유튜브 제목** 사용(금지문자 제거/윈도우 예약어 회피/길이 제한)
# - UI: 사이버펑크 테마, 세션 유지, 큰 '전체 ZIP' 버튼 + meta.txt
# - 안정성: 더 강한 예외 처리/재시도/시간 제한/메모리 보호/프레임 워커 제한
# - 유틸: 번들 초기화 버튼, meta.txt 단독 다운로드, DeepL 쿼터/429 표시, 환경 경로 힌트
# ------------------------------------------------------------------------------------------------

import os, re, io, time, tempfile, zipfile, traceback, random, json, datetime
from typing import List, Dict, Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps
from rapidfuzz import fuzz as rf_fuzz
import requests

# ---------- 존재여부 체크 ----------
HAS_YTA = True
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    HAS_YTA = False

HAS_WHISPER = True
try:
    import whisper
except Exception:
    HAS_WHISPER = False

HAS_YTDLP = True
try:
    from yt_dlp import YoutubeDL
except Exception:
    HAS_YTDLP = False

HAS_FFMPEG = True
try:
    import ffmpeg
except Exception:
    HAS_FFMPEG = False

HAS_TESS = True
try:
    import pytesseract
except Exception:
    HAS_TESS = False

HAS_ARGOS = True
try:
    import argostranslate.package, argostranslate.translate
except Exception:
    HAS_ARGOS = False

# =========================
# App Config + Theme
# =========================
st.set_page_config(page_title="쇼츠 카피 작업실", layout="wide")
st.markdown(
    """
<style>
:root{--bg0:#0a0b10;--bg1:#121423;--neon1:#00e5ff;--neon2:#ff00e5;--neon3:#7cff00;--text:#e6f7ff}
html,body,.stApp{background:radial-gradient(60% 80% at 50% 20%, rgba(0,229,255,.15), transparent 60%),
                 radial-gradient(50% 50% at 80% 10%, rgba(255,0,229,.10), transparent 50%),
                 linear-gradient(180deg,var(--bg1) 0%,var(--bg0) 100%);color:var(--text)}
h1,h2,h3,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3{ text-shadow:0 0 8px rgba(0,229,255,.5),0 0 16px rgba(255,0,229,.3)}
section[data-testid="stSidebar"]{background:linear-gradient(180deg, rgba(10,11,16,.9), rgba(18,20,35,.9));border-right:1px solid rgba(0,229,255,.25)}
div[data-testid="stStatusWidget"],.stProgress>div>div>div{background:linear-gradient(90deg, rgba(0,229,255,.3), rgba(255,0,229,.3))!important}
.stButton>button,.stDownloadButton>button{border:1px solid rgba(0,229,255,.55);box-shadow:0 0 10px rgba(0,229,255,.35), inset 0 0 10px rgba(0,229,255,.12);
background:rgba(0,0,0,.35);color:var(--text)}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:rgba(255,0,229,.7);box-shadow:0 0 12px rgba(255,0,229,.45), inset 0 0 12px rgba(255,0,229,.18)}
[data-baseweb="select"]>div{border-color:rgba(0,229,255,.4)!important}
textarea,input,.stTextInput>div>div>input{background:rgba(255,255,255,.04)!important;color:var(--text)!important;border:1px solid rgba(0,229,255,.25)!important}
hr,.stDivider{border-top:1px dashed rgba(124,255,0,.4)!important}
.bigzip button{font-size:18px;padding:12px 18px;border-width:2px}
.smallwarn{opacity:.8;font-size:12px}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎬 쇼츠 카피 작업실")
st.caption("CC → Whisper → (옵션) OCR 추출 후, DeepL 우선 번역 / Argos 폴백. 연구·참고용만.")

# =========================
# Sidebar Settings (세션 유지)
# =========================
with st.sidebar:
    st.header("⚙️ 추출 설정")
    pref_langs = st.multiselect(
        "우선 시도할 CC 자막 언어",
        ["ko","en","ja","zh-Hans","zh-Hant"],
        default=st.session_state.get("pref_langs", ["ko","en"]),
    )
    st.session_state["pref_langs"] = pref_langs

    whisper_size = st.selectbox(
        "Whisper 모델",
        ["tiny","base","small","medium"],
        index={"tiny":0,"base":1,"small":2,"medium":3}.get(st.session_state.get("whisper_size","small"),2),
        help="설치/환경에 따라 비활성될 수 있어요.",
    )
    st.session_state["whisper_size"] = whisper_size

    do_ocr = st.checkbox("OCR로 하드자막도 추출", value=st.session_state.get("do_ocr", False))
    st.session_state["do_ocr"] = do_ocr

    ocr_fps = st.slider("OCR 프레임 추출 FPS", 0.5, 4.0, st.session_state.get("ocr_fps", 1.0), 0.5)
    st.session_state["ocr_fps"] = ocr_fps

    ocr_lang = st.text_input("Tesseract 언어 코드", st.session_state.get("ocr_lang","kor+eng"))
    st.session_state["ocr_lang"] = ocr_lang

    default_workspace = st.text_input("기본 작업 경로(메모)", st.session_state.get("default_workspace", r"E:\\Youtube\\쇼츠카피 작업실"))
    st.session_state["default_workspace"] = default_workspace

    win_tess = st.text_input("Windows Tesseract 경로(옵션)", st.session_state.get("win_tess", r"E:\\Youtube\\쇼츠카피 작업실\\Tesseract-OCR\\tesseract.exe"))
    st.session_state["win_tess"] = win_tess

    crop_bottom = st.checkbox("하단 40%만 OCR(자막 집중)", value=st.session_state.get("crop_bottom", True))
    st.session_state["crop_bottom"] = crop_bottom

    sim_threshold = st.slider("OCR 중복 합치기 유사도(%)", 60, 95, st.session_state.get("sim_threshold", 80))
    st.session_state["sim_threshold"] = sim_threshold

    include_srt = st.checkbox("원문 SRT 생성", value=st.session_state.get("include_srt", True))
    st.session_state["include_srt"] = include_srt

    st.divider()
    st.header("🌍 번역 설정")
    st.caption("DeepL 우선 → 오류/쿼터 시 Argos(설치 시) 폴백. 긴 텍스트 자동 분할.")
    target_langs = st.multiselect(
        "번역할 언어",
        ["English (EN-US)", "Japanese (JA)", "Chinese - Simplified (ZH)", "Hindi (HI)", "Spanish (ES)", "Arabic (AR)"],
        default=st.session_state.get(
            "target_langs",
            ["English (EN-US)", "Japanese (JA)", "Chinese - Simplified (ZH)", "Spanish (ES)"],
        ),
    )
    st.session_state["target_langs"] = target_langs

    manual_src = st.selectbox(
        "소스 언어(모름이면 자동 추정)",
        ["auto","ko","en","ja","zh","es","ar","hi"],
        index=["auto","ko","en","ja","zh","es","ar","hi"].index(st.session_state.get("manual_src","auto")),
    )
    st.session_state["manual_src"] = manual_src

    st.divider()
    st.header("📝 TXT 옵션")
    vrew_max_chars = st.slider("Vrew TXT 줄당 최대 글자(0=제한없음)", 0, 40, st.session_state.get("vrew_max_chars", 0))
    st.session_state["vrew_max_chars"] = vrew_max_chars

    vrew_strip_emoji = st.checkbox("이모지/이모티콘 제거", value=st.session_state.get("vrew_strip_emoji", True))
    st.session_state["vrew_strip_emoji"] = vrew_strip_emoji

# Keys / Paths
DEEPL_KEY = st.secrets.get("DEEPL_API_KEY", os.getenv("DEEPL_API_KEY",""))
DEEPL_KEY = (DEEPL_KEY or "").strip()
if os.name == "nt" and HAS_TESS and win_tess and os.path.exists(win_tess):
    try:
        pytesseract.pytesseract.tesseract_cmd = win_tess
    except Exception:
        pass

# =========================
# Helpers: Common
# =========================
RESERVED_WIN = {"CON","PRN","AUX","NUL","COM1","COM2","COM3","COM4","COM5","COM6","COM7","COM8","COM9","LPT1","LPT2","LPT3","LPT4","LPT5","LPT6","LPT7","LPT8","LPT9"}

def _requests_session():
    s = requests.Session()
    s.headers.update({"User-Agent":"ShortsCopyLab/1.4 (+streamlit)"})
    s.timeout = 20
    return s
SESS = _requests_session()


def extract_video_id(url: str) -> Optional[str]:
    if not url: return None
    m = re.search(r'(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None


def safe_filename(name: str, limit: int = 110) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # 윈도우 예약어 회피
    if name.split(".")[0].upper() in RESERVED_WIN:
        name = f"_{name}"
    # 끝 공백/점을 제거
    name = name[:limit].rstrip(" .")
    if not name:
        name = f"untitled_{int(time.time())}"
    return name


def get_title_by_ytdlp(url: str) -> Optional[str]:
    if not HAS_YTDLP: return None
    try:
        ydl_opts = {"quiet": True, "skip_download": True}
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title")
    except Exception:
        return None


def get_title_by_oembed(url: str) -> Optional[str]:
    try:
        r = SESS.get("https://www.youtube.com/oembed", params={"url": url, "format":"json"}, timeout=10)
        if r.status_code == 200:
            return r.json().get("title")
    except Exception:
        pass
    return None


def get_video_title(url: str, vid: str) -> str:
    for fn in (get_title_by_ytdlp, get_title_by_oembed):
        t = fn(url)
        if t: return safe_filename(t)
    return vid


@st.cache_resource(show_spinner=False)
def load_whisper(model_name: str):
    if not HAS_WHISPER:
        raise RuntimeError("Whisper 미설치")
    return whisper.load_model(model_name)


@st.cache_data(show_spinner=False, ttl=3600)
def try_cc_transcript(video_id: str, langs: List[str]) -> Optional[List[Dict]]:
    if not HAS_YTA: return None
    try:
        for L in langs:
            try:
                t = YouTubeTranscriptApi.get_transcript(video_id, languages=[L])
                if t: return t
            except Exception:
                pass
        # 자동 생성/자동 번역 자막도 포함한 리스트에서 첫 항목
        try:
            avail = YouTubeTranscriptApi.list_transcripts(video_id)
            for tr in avail:
                try:
                    t = tr.fetch()
                    if t: return t
                except Exception:
                    pass
        except Exception:
            pass
        return None
    except Exception:
        return None


def ytdlp_download(url: str, outdir: str, audio_only=True) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    if not HAS_YTDLP: return None, None, None
    audio_path = None; video_path = None; info_obj = None
    ydl_opts = {
        "outtmpl": f"{outdir}/%(id)s.%(ext)s",
        "quiet": True, "nocheckcertificate": True, "retries": 3, "fragment_retries": 3,
    }
    ydl_opts["format"] = "bestaudio/best" if audio_only else "bestvideo[ext=mp4]+bestaudio/best/best"
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            info_obj = info
            base_id = info.get("id")
            for f in os.listdir(outdir):
                if not f.startswith(base_id + "."): continue
                ext = f.split(".")[-1].lower()
                p = os.path.join(outdir, f)
                if ext in ("m4a","webm","mp3","opus","ogg","aac","wav"):
                    audio_path = p
                elif ext in ("mp4","mkv","webm","mov"):
                    video_path = p
    except Exception:
        return None, None, None
    return audio_path, video_path, info_obj


def hms_from_seconds(sec: float) -> str:
    ms = int((sec - int(sec)) * 1000); total = int(sec)
    h = total // 3600; m = (total % 3600) // 60; s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_txt_plain_from_segments(segs: List[Dict]) -> str:
    return "\n".join([s.get("text","") for s in segs if s.get("text")])


def build_txt_timecoded_from_segments(segs: List[Dict]) -> str:
    return "\n".join([f"[{int(round(s['start'])):>4d}s] {s['text']}" for s in segs if s.get("text")])


def build_srt_from_segments(segs: List[Dict]) -> str:
    chunks=[]
    for i, s in enumerate(segs, 1):
        chunks.append(f"{i}\n{hms_from_seconds(float(s['start']))} --> {hms_from_seconds(float(s['end']))}\n{s.get('text','')}\n")
    return "\n".join(chunks).strip()

# OCR helpers

def extract_frames_to_dir(video_path: str, outdir: str, fps: float):
    if not HAS_FFMPEG: return [], fps
    pattern = os.path.join(outdir, "frame_%06d.jpg")
    try:
        ffmpeg.input(video_path).output(pattern, **{"vf":f"fps={fps}"}, qscale=2, loglevel="error").overwrite_output().run()
        frames = [os.path.join(outdir, f) for f in sorted(os.listdir(outdir)) if f.startswith("frame_")]
        return frames, fps
    except Exception:
        return [], fps


def crop_bottom_region(img: Image.Image, ratio: float = 0.4) -> Image.Image:
    w,h = img.size; top = int(h*(1.0 - ratio))
    return img.crop((0, top, w, h))


def ocr_image(img: Image.Image, lang: str) -> str:
    if not HAS_TESS: return ""
    gray = ImageOps.grayscale(img)
    try: return pytesseract.image_to_string(gray, lang=lang).strip()
    except Exception: return ""


def dedup_ocr_timecoded(records: List[Tuple[int,str]], threshold: int) -> List[Tuple[int,str]]:
    cleaned=[]; last_txt=""
    for t, txt in records:
        if not txt: continue
        compact=" ".join(txt.split()); 
        if not compact: continue
        if last_txt and rf_fuzz.ratio(compact, last_txt) >= threshold: 
            continue
        cleaned.append((t, compact)); last_txt = compact
    return cleaned


def build_txt_timecoded_from_ocr(ocr_records: List[Tuple[int,str]]) -> str:
    return "\n".join([f"[{t:>4d}s] {txt}" for t, txt in ocr_records])


def build_plain_from_ocr(ocr_records: List[Tuple[int,str]]) -> str:
    return "\n".join([txt for _, txt in ocr_records])

# Lang guess

def guess_lang_code(text: str) -> str:
    if re.search(r"[\u3131-\uD79D]", text): return "ko"
    if re.search(r"[\u3040-\u30ff]", text): return "ja"
    if re.search(r"[\u4e00-\u9fff]", text): return "zh"
    if re.search(r"[\u0600-\u06FF]", text): return "ar"
    if re.search(r"[\u0900-\u097F]", text): return "hi"
    if re.search(r"\b(el|la|de|que|y|en|los|del)\b", text.lower()): return "es"
    return "en"

# Translation (DeepL → Argos)
DEEPL_TARGET_MAP = {
    "English (EN-US)": "EN-US",
    "Japanese (JA)": "JA",
    "Chinese - Simplified (ZH)": "ZH",
    "Hindi (HI)": "HI",
    "Spanish (ES)": "ES",
    "Arabic (AR)": "AR",
}
ARGOS_CODE_MAP = {
    "English (EN-US)": "en",
    "Japanese (JA)": "ja",
    "Chinese - Simplified (ZH)": "zh",
    "Hindi (HI)": "hi",
    "Spanish (ES)": "es",
    "Arabic (AR)": "ar",
}


def _retry_post(url, data, tries=2, backoff=0.7, timeout=60):
    for i in range(tries):
        try:
            r = SESS.post(url, data=data, timeout=timeout)
            if r.status_code == 200: 
                return True, r
            # 429/456 등은 바로 알려주기
            if r.status_code in (429, 456):
                return False, r
        except Exception:
            pass
        if i < tries-1: 
            time.sleep(backoff*(1+i) + random.random()*0.2)
    return False, None


def _deepl_post(texts: List[str], target: str, source: Optional[str]=None) -> Tuple[bool, List[str], str]:
    if not DEEPL_KEY: return (False, [], "DEEPL_API_KEY not set")
    payload = []; [payload.append(("text", t)) for t in texts]
    data = [("auth_key", DEEPL_KEY), ("target_lang", target)] + payload
    if source and source != "auto": data.append(("source_lang", source.upper()))
    for url in ["https://api-free.deepl.com/v2/translate", "https://api.deepl.com/v2/translate"]:
        ok, resp = _retry_post(url, data=data, tries=2, backoff=0.7, timeout=60)
        if ok and resp is not None:
            try:
                res = resp.json()
                outs = [t["text"] for t in res.get("translations", [])]
                if len(outs) != len(texts): 
                    return (False, [], f"DeepL mismatch: {len(outs)} vs {len(texts)}")
                return (True, outs, "")
            except Exception as e:
                return (False, [], f"DeepL parse error: {e}")
        # 429/456 표시
        if resp is not None and resp.status_code in (429,456):
            try:
                detail = resp.json()
            except Exception:
                detail = {"message":"Rate limited or quota exceeded"}
            return (False, [], f"DeepL HTTP {resp.status_code}: {detail}")
    return (False, [], "DeepL error or quota/plan issue at both endpoints")


def chunk_text_by_chars(text: str, max_chars: int = 4800) -> List[str]:
    if len(text) <= max_chars: return [text]
    blocks, buf = [], ""
    for line in text.split("\n"):
        parts = re.split(r'(?<=[\.!\?])\s+', line)
        for p in parts:
            if not p: continue
            if len(buf) + len(p) + 1 <= max_chars:
                buf = (buf + "\n" + p).strip()
            else:
                if buf: blocks.append(buf)
                while len(p) > max_chars:
                    blocks.append(p[:max_chars]); p = p[max_chars:]
                buf = p
    if buf: blocks.append(buf)
    return blocks


def ensure_argos_pair(src_code: str, tgt_code: str) -> bool:
    if not HAS_ARGOS: return False
    try:
        installed = argostranslate.translate.get_installed_languages()
        for L in installed:
            if L.code == src_code:
                for T in L.translations:
                    if T.to_lang.code == tgt_code: return True
        available = argostranslate.package.get_available_packages()
        cand = [p for p in available if p.from_code == src_code and p.to_code == tgt_code]
        if cand:
            with tempfile.TemporaryDirectory() as td:
                path = cand[0].download(td)
                argostranslate.package.install_from_path(path)
            return True
        return False
    except Exception:
        return False


def argos_translate_text(text: str, src_code: str, tgt_code: str) -> Tuple[bool, str, str]:
    if not HAS_ARGOS: return (False, "", "Argos not installed")
    try:
        ok = ensure_argos_pair(src_code, tgt_code)
        if not ok: return (False, "", f"Argos pair not installed: {src_code}->{tgt_code}")
        installed = argostranslate.translate.get_installed_languages()
        src = next((L for L in installed if L.code == src_code), None)
        if not src: return (False, "", f"Argos missing source {src_code}")
        trans = next((T for T in src.translations if T.to_lang.code == tgt_code), None)
        if not trans: return (False, "", f"Argos missing pair {src_code}->{tgt_code}")
        return (True, trans.translate(text), "")
    except Exception as e:
        return (False, "", f"Argos exception: {e}")


def smart_translate_text(text: str, target_label: str, src_hint: str="auto") -> Tuple[str, str]:
    if not text.strip(): return "", "empty"
    target_deepl = DEEPL_TARGET_MAP[target_label]; target_argos = ARGOS_CODE_MAP[target_label]
    src_code = src_hint if src_hint!="auto" else guess_lang_code(text)
    deepl_src = None if src_hint=="auto" else src_hint.upper()
    if DEEPL_KEY:
        chunks = chunk_text_by_chars(text, max_chars=4800)
        ok, outs, msg = _deepl_post(chunks, target_deepl, source=deepl_src)
        if ok: return "\n".join(outs), "DeepL"
        else:
            st.info(f"DeepL 사용 불가 → Argos 폴백 시도 ({msg})", icon="⚠️")
    ok2, out2, _ = argos_translate_text(text, src_code, target_argos)
    if ok2: return out2, "Argos"
    if target_argos != "en":
        ok3, mid, _ = argos_translate_text(text, src_code, "en")
        if ok3:
            ok4, out4, _ = argos_translate_text(mid, "en", target_argos)
            if ok4: return out4, "Argos(pivot)"
    return "[번역 실패: DeepL 미사용/오류, Argos 미설치 또는 오류]", "Failed"


def translate_segments_to_srt(segs: List[Dict], target_label: str, src_hint: str="auto") -> Tuple[str, str]:
    target_deepl = DEEPL_TARGET_MAP[target_label]
    lines = [s.get("text","") for s in segs]; backend_used = None
    if DEEPL_KEY:
        out_lines = []; batch = 50
        for i in range(0, len(lines), batch):
            chunk = lines[i:i+batch]; expanded=[]; idx_map=[]
            for j, line in enumerate(chunk):
                parts = chunk_text_by_chars(line, max_chars=4800)
                expanded.extend(parts); idx_map.append((len(parts), j))
            ok, outs, msg = _deepl_post(expanded, target_deepl, source=None if src_hint=="auto" else src_hint.upper())
            if not ok: 
                out_lines = None; 
                st.info(f"SRT 번역 DeepL 실패 → Argos 라인별 폴백 ({msg})", icon="⚠️")
                break
            rec=[]; k=0
            for cnt,_ in idx_map:
                merged="\n".join(outs[k:k+cnt]); k+=cnt; rec.append(merged)
            out_lines.extend(rec); backend_used="DeepL"
        if out_lines is not None and len(out_lines)==len(lines):
            srt_chunks=[]
            for idx,(s,tline) in enumerate(zip(segs, out_lines),1):
                srt_chunks.append(f"{idx}\n{hms_from_seconds(float(s['start']))} --> {hms_from_seconds(float(s['end']))}\n{tline}\n")
            return "\n".join(srt_chunks).strip(), backend_used
    # Argos 라인별
    out_lines=[]; any_argos=False
    for ln in lines:
        ttxt, bk = smart_translate_text(ln, target_label, src_hint=src_hint)
        out_lines.append(ttxt); any_argos = any_argos or bk.startswith("Argos"); backend_used = backend_used or bk
    if not any_argos and backend_used is None: backend_used="Failed"
    srt_chunks=[]
    for idx,(s,tline) in enumerate(zip(segs, out_lines),1):
        srt_chunks.append(f"{idx}\n{hms_from_seconds(float(s['start']))} --> {hms_from_seconds(float(s['end']))}\n{tline}\n")
    return "\n".join(srt_chunks).strip(), backend_used or "Argos/Failed"

# Vrew / one-line helpers
EMOJI_RE = re.compile("[" "\U0001F300-\U0001F6FF" "\U0001F900-\U0001F9FF" "\U0001F1E6-\U0001F1FF" "\u2600-\u27BF" "]+", flags=re.UNICODE)

def clean_for_vrew(text: str, strip_emoji: bool=True) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return EMOJI_RE.sub("", t) if strip_emoji else t


def split_to_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!\?]|[。！？]|[…]|[~]|[♪]|[”»])\s+", text)
    return [p.strip(" \"'“”‘’»«") for p in parts if p and p.strip()]


def wrap_kchars(s: str, k: int) -> List[str]:
    if k <= 0 or len(s) <= k: return [s]
    out=[]; buf=""
    for ch in s:
        if len(buf)+len(ch) <= k: buf += ch
        else: out.append(buf); buf = ch
    if buf: out.append(buf)
    return out


def build_vrew_txt_from_segments(segs: List[Dict], max_chars:int=0, strip_emoji:bool=True) -> str:
    lines=[]
    for s in segs:
        t = clean_for_vrew(s.get("text",""), strip_emoji=strip_emoji)
        for sent in split_to_sentences(t):
            if not sent: continue
            wrapped = wrap_kchars(sent, max_chars) if max_chars>0 else [sent]
            lines.extend(wrapped)
    return "\n".join(lines).strip()


def build_timecoded_one_line_txt(segs: List[Dict]) -> str:
    lines=[]
    for s in segs:
        start = hms_from_seconds(float(s["start"])); end = hms_from_seconds(float(s["end"]))
        text  = re.sub(r"\s+", " ", s.get("text",""))
        text  = text.strip()
        lines.append(f"{start} --> {end} | {text}")
    return "\n".join(lines)

# =========================
# Session State for bundle
# =========================
if "bundle" not in st.session_state:
    st.session_state.bundle = {}   # { filename: bytes }
if "meta" not in st.session_state:
    st.session_state.meta = {}     # info for meta.txt
if "last" not in st.session_state:
    st.session_state.last = {}     # url, vid, title


def add_to_bundle(fname: str, data: bytes):
    # 동일 파일명 덮어쓰기 허용(가장 최근 결과)
    st.session_state.bundle[fname] = data


def reset_bundle():
    st.session_state.bundle = {}
    st.session_state.meta = {}


def build_zip_bytes() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        # meta
        meta_txt = json.dumps(st.session_state.meta, ensure_ascii=False, indent=2)
        zf.writestr("meta.txt", meta_txt.encode("utf-8"))
        # files
        for fname, data in st.session_state.bundle.items():
            zf.writestr(fname, data)
    return bio.getvalue()

# =========================
# UI — Main
# =========================
colA, colB = st.columns([4,1])
with colA:
    url = st.text_input(
        "🔗 유튜브 쇼츠 URL",
        placeholder="https://youtube.com/shorts/xxxxxxxxxxx",
        value=st.session_state.last.get("url",""),
    )
with colB:
    if st.button("🧹 번들 초기화", use_container_width=True):
        reset_bundle()
        st.success("현재 번들을 초기화했어요.")

run = st.button("🚀 추출 시작", type="primary")

with st.expander("환경 체크"):
    st.write(f"- youtube-transcript-api: {'✅' if HAS_YTA else '⛔'}")
    st.write(f"- Whisper: {'✅' if HAS_WHISPER else '⛔'}")
    st.write(f"- ffmpeg-python: {'✅' if HAS_FFMPEG else '⛔'}")
    st.write(f"- yt-dlp: {'✅' if HAS_YTDLP else '⛔'}")
    st.write(f"- Tesseract(pytesseract): {'✅' if HAS_TESS else '⛔'}")
    st.write(f"- Argos: {'✅' if HAS_ARGOS else '⛔'}")
    st.write(f"- DeepL Key: {'✅' if bool(DEEPL_KEY) else '⛔'}")
    st.markdown(
        "<div class='smallwarn'>※ Windows에서 ffmpeg, Tesseract 경로 미설치시: ffmpeg.exe PATH 등록, Tesseract 경로 위 입력에 지정하세요.</div>",
        unsafe_allow_html=True,
    )

if run:
    reset_bundle()
    if not url:
        st.error("URL을 입력해주세요."); st.stop()
    vid = extract_video_id(url)
    if not vid:
        st.error("유효한 유튜브/쇼츠 URL이 아닙니다."); st.stop()

    title = get_video_title(url, vid)
    st.session_state.last = {"url": url, "vid": vid, "title": title}

    with st.status("처리 중...", expanded=True) as status:
        try:
            st.write("1) CC 자막 확인 중…")
            cc = try_cc_transcript(vid, pref_langs) if HAS_YTA else None

            ocr_time = ""; ocr_plain = ""
            segs: List[Dict] = []
            if cc:
                st.success("CC 자막을 가져왔습니다.")
                for s in cc:
                    start=float(s.get("start",0.0))
                    end = start + max(float(s.get("duration",0.01)),0.01)
                    segs.append({"start":start,"end":end,"text":s.get("text","").strip()})
                source_origin = "CC"
            else:
                if not HAS_WHISPER or not HAS_YTDLP:
                    st.error("CC 없음 + Whisper/yt-dlp 미가용 → 음성 인식 불가.\n- youtube-transcript-api 또는 Whisper/yt-dlp 설치/활성화 필요")
                    st.stop()
                with tempfile.TemporaryDirectory() as td:
                    st.write("2) 오디오 다운로드(yt-dlp)…")
                    audio_path, _video_path, info_obj = ytdlp_download(url, td, audio_only=True)
                    # 제목 보정(yt-dlp info에 더 정확한 제목이 있으면 대체)
                    if info_obj and info_obj.get("title"):
                        title = safe_filename(info_obj["title"])
                        st.session_state.last["title"] = title
                    if not audio_path: 
                        st.error("오디오 다운로드 실패"); st.stop()
                    st.write(f"3) Whisper({whisper_size}) 음성 인식…")
                    model = load_whisper(whisper_size)
                    result = model.transcribe(audio_path, verbose=False, fp16=False)
                    for s in result.get("segments", []):
                        segs.append({"start":float(s["start"]),"end":float(s["end"]),"text":s.get("text","").strip()})
                source_origin = "Whisper"

            # 원문 가공물
            src_text_plain   = build_txt_plain_from_segments(segs)
            src_text_time    = build_txt_timecoded_from_segments(segs)
            src_vrew_txt     = build_vrew_txt_from_segments(segs, max_chars=vrew_max_chars, strip_emoji=vrew_strip_emoji)
            src_timecoded_1l = build_timecoded_one_line_txt(segs)
            srt_text         = build_srt_from_segments(segs) if include_srt else ""

            # (옵션) OCR — CC/Whisper 모두 실패했을 때만 시도하도록 유지
            if (not cc) and do_ocr:
                if not (HAS_TESS and HAS_YTDLP and HAS_FFMPEG):
                    st.warning("OCR은 pytesseract/yt-dlp/ffmpeg 필요(자동 건너뜀)")
                else:
                    with tempfile.TemporaryDirectory() as td2:
                        st.write("4) OCR용 영상/프레임 추출…")
                        _, video_path, _ = ytdlp_download(url, td2, audio_only=False)
                        if video_path:
                            frames_dir = os.path.join(td2, "frames"); os.makedirs(frames_dir, exist_ok=True)
                            frame_paths, _ = extract_frames_to_dir(video_path, frames_dir, ocr_fps)
                            ocr_records=[]; progress = st.progress(0.0, text="OCR 진행 중…"); total = len(frame_paths)
                            for idx, fp in enumerate(frame_paths):
                                try:
                                    with Image.open(fp) as im:
                                        if crop_bottom: im = crop_bottom_region(im, 0.4)
                                        txt = ocr_image(im, ocr_lang)
                                    sec = int(idx/ocr_fps)
                                    if txt: ocr_records.append((sec, txt))
                                except Exception:
                                    pass
                                if total: progress.progress((idx+1)/total)
                            ded = dedup_ocr_timecoded(ocr_records, sim_threshold)
                            ocr_time = build_txt_timecoded_from_ocr(ded)
                            ocr_plain = build_plain_from_ocr(ded)
                        else:
                            st.error("영상 다운로드 실패(OCR 건너뜀)")
                    if ocr_plain:
                        st.info("🔤 OCR(하드자막)도 추출됨")
                        st.text_area("OCR 타임코드 미리보기", ocr_time[:2000], height=150)

            status.update(label="대본 추출 완료 ✅", state="complete", expanded=False)

        except Exception as e:
            st.error(f"처리 중 오류: {e}")
            st.code(traceback.format_exc()); st.stop()

    # ===== 원문 다운로드 & 묶음에 추가 =====
    st.subheader("📜 원문 대본")
    st.caption(f"원본 소스: {source_origin} · 제목: {title}")
    st.text_area("타임코드 미리보기", src_text_time[:4000], height=200)

    base = title or st.session_state.last["vid"]
    files_now = []

    # 파일 생성 및 번들 적재
    files_now.append( (f"{base}_plain.txt",               src_text_plain.encode("utf-8")) )
    files_now.append( (f"{base}_timecoded_block.txt",     src_text_time.encode("utf-8")) )
    files_now.append( (f"{base}_vrew.txt",                src_vrew_txt.encode("utf-8")) )
    files_now.append( (f"{base}_timecoded_oneline.txt",   src_timecoded_1l.encode("utf-8")) )
    if include_srt:
        files_now.append( (f"{base}.srt",                 srt_text.encode("utf-8")) )
    if 'ocr_plain' in locals() and ocr_plain:
        files_now.append( (f"{base}_ocr_plain.txt",       ocr_plain.encode("utf-8")) )
        files_now.append( (f"{base}_ocr_timecoded.txt",   ocr_time.encode("utf-8")) )

    # 개별 다운로드 버튼
    cols = st.columns(6)
    for i,(fname,data) in enumerate(files_now):
        add_to_bundle(fname, data)
        with cols[i % 6]:
            st.download_button(f"💾 {os.path.splitext(fname)[0].split('_')[-1]}",
                               data, file_name=fname, key=f"dl_src_{i}")

    # meta 갱신
    st.session_state.meta.update({
        "url": url,
        "video_id": st.session_state.last["vid"],
        "title": title,
        "source_origin": source_origin,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "workspace_note": default_workspace,
        "selected_translation_targets": target_langs,
        "backends": {}  # 언어별 번역 백엔드 기록 예정
    })

    st.divider()

    # ===== 번역 섹션 =====
    st.subheader("🌍 다국어 번역")
    src_hint = manual_src

    if st.button("🧠 번역 생성 (DeepL 우선 → Argos 폴백)"):
        base_segs = segs
        base_plain = build_txt_plain_from_segments(base_segs)

        with st.status("번역 중…", expanded=True) as tstat:
            used_backends = {}
            for label in target_langs:
                lang_code = ARGOS_CODE_MAP.get(label, "xx")

                # 평문 TXT
                out_text, backend = smart_translate_text(base_plain, label, src_hint=src_hint)
                used_backends[label] = backend
                fname_txt = f"{base}_{lang_code}.txt"
                add_to_bundle(fname_txt, out_text.encode("utf-8"))
                st.download_button(f"💾 {label} TXT", out_text, file_name=fname_txt, key=f"dl_txt_{lang_code}")

                # Vrew TXT
                vrew_lines=[]
                for sent in split_to_sentences(out_text):
                    for w in (wrap_kchars(sent, vrew_max_chars) if vrew_max_chars>0 else [sent]):
                        vrew_lines.append(clean_for_vrew(w, strip_emoji=vrew_strip_emoji))
                vrew_txt="\n".join([l for l in vrew_lines if l.strip()])
                fname_vrew = f"{base}_{lang_code}_vrew.txt"
                add_to_bundle(fname_vrew, vrew_txt.encode("utf-8"))
                st.download_button(f"💾 {label} Vrew TXT", vrew_txt, file_name=fname_vrew, key=f"dl_vrew_{lang_code}")

                # 타임코드 TXT(한 줄형) — 세그먼트별 번역
                timecoded_lines=[]
                for s in base_segs:
                    seg_txt = s.get("text","")
                    ttxt, _bk = smart_translate_text(seg_txt, label, src_hint=src_hint)
                    timecoded_lines.append(f"{hms_from_seconds(float(s['start']))} --> {hms_from_seconds(float(s['end']))} | {ttxt}")
                timecoded_one = "\n".join(timecoded_lines)
                fname_oneline = f"{base}_{lang_code}_timecoded_oneline.txt"
                add_to_bundle(fname_oneline, timecoded_one.encode("utf-8"))
                st.download_button(f"💾 {label} 타임코드 TXT", timecoded_one, file_name=fname_oneline, key=f"dl_oneline_{lang_code}")

                # SRT
                srt_tr, backend3 = translate_segments_to_srt(base_segs, label, src_hint=src_hint)
                st.write(f"  · {label} SRT 백엔드: {backend3}")
                fname_srt = f"{base}_{lang_code}.srt"
                add_to_bundle(fname_srt, srt_tr.encode("utf-8"))

            st.session_state.meta["backends"] = {label: used_backends.get(label, "") for label in target_langs}
            if not DEEPL_KEY:
                st.info("ℹ️ DeepL 키가 없어 Argos(설치 시)에만 의존합니다.")
            elif any(str(used_backends.get(k,"")).startswith("Argos") for k in used_backends):
                st.info("ℹ️ 일부 언어에서 DeepL 제한/오류로 Argos 폴백을 사용했습니다.")

            tstat.update(label="번역 완료 ✅", state="complete", expanded=False)

# ===== 항상 표시: 전체 ZIP / meta 단독 =====
st.divider()
if st.session_state.bundle:
    zip_bytes = build_zip_bytes()
    st.download_button(
        "⬇️ 전체 다운로드 (ZIP)",
        data=zip_bytes,
        file_name=f"{(st.session_state.last.get('title') or st.session_state.last.get('vid') or 'shorts')}_all_outputs.zip",
        key="big_zip",
        type="primary",
        help="현재 화면에 생성된 원문/번역/OCR 등 전체를 한 번에 저장합니다.",
        use_container_width=True,
    )

    # meta.txt 단독 다운로드도 지원
    meta_txt = json.dumps(st.session_state.meta, ensure_ascii=False, indent=2)
    st.download_button(
        "📝 meta.txt만 저장",
        data=meta_txt.encode("utf-8"),
        file_name="meta.txt",
        key="meta_only",
    )
else:
    st.info("생성된 파일이 없습니다. URL을 넣고 처리/번역을 먼저 실행하세요.")

st.divider()
st.caption("ⓘ 브라우저 다운로드 폴더로 저장됩니다. 시스템 경로를 강제 지정할 수는 없어요. 저작권/플랫폼 약관을 준수하세요.")
