import io
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_OK = True
except Exception:
    PDF2IMAGE_OK = False

try:
    import fitz
    PYMUPDF_OK = True
except Exception:
    PYMUPDF_OK = False

try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = False
except Exception:
    CROPPER_OK = False


APP_NAME = "HEY ROI"
APP_VERSION = "v10"
AUTHOR_LINE = "by Dr. Cevdet ZÜNGÜN"
INTERPRETATION_SIGNATURE = "by Dr. Bağdagül ÇAKIR"
USERS_FILE = Path("hey_roi_users.json")
LOG_FILE = Path("hey_roi_activity_log.csv")

DEFAULT_PEAK_ORDER = [
    "Unknown",
    "A1a",
    "A1b",
    "F",
    "LA1c/CHb-1",
    "A1c",
    "P3",
    "A0",
    "A2",
    "S-Window",
]

st.set_page_config(
    page_title=f"{APP_NAME} - Hemoglobin Elektroforez ROI Yorumlayıcı",
    layout="wide"
)


# -----------------------------
# User management
# -----------------------------
def ensure_users_file():
    if not USERS_FILE.exists():
        default_users = {
            "admin": {"password": "admin123", "role": "admin"}
        }
        USERS_FILE.write_text(
            json.dumps(default_users, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


def load_users() -> dict:
    ensure_users_file()
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"admin": {"password": "admin123", "role": "admin"}}


def save_users(users: dict):
    USERS_FILE.write_text(
        json.dumps(users, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# -----------------------------
# Log management
# -----------------------------
def ensure_log_file():
    if not LOG_FILE.exists():
        df = pd.DataFrame(columns=[
            "timestamp",
            "username",
            "role",
            "action",
            "details",
        ])
        df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")


def write_log(action: str, details: str = ""):
    ensure_log_file()
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": st.session_state.get("username", ""),
        "role": st.session_state.get("role", ""),
        "action": action,
        "details": details,
    }])
    row.to_csv(
        LOG_FILE,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig"
    )


def read_logs() -> pd.DataFrame:
    ensure_log_file()
    try:
        return pd.read_csv(LOG_FILE, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=["timestamp", "username", "role", "action", "details"])


def login_screen():
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 100%);
                border: 1px solid #dbe7ff;
                border-radius: 18px;
                padding: 26px 28px 18px 28px;
                box-shadow: 0 8px 24px rgba(44, 62, 80, 0.08);
                margin-top: 40px;
                ">
                <div style="text-align: center; margin: 0 0 8px 0; color: #1f3c88; font-size: 1.7rem; font-weight: 700;">
                    HEY-ROI (by Dr. Cevdet ZÜNGÜN)
                </div>
                <div style="text-align: center; margin: 0 0 10px 0; color: #1f3c88; font-size: 1.35rem; font-weight: 600;">
                    Giriş
                </div>
                <div style="font-size: 0.95rem; color: #4b587c; margin-bottom: 6px; text-align: center;">
                    Hemoglobin elektroforez ROI yorum destek sistemi
                </div>
                <div style="font-size: 0.9rem; color: #6c7a99; text-align: center;">
                    Yetkili kullanıcı girişi gereklidir.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Kullanıcı adı")
            password = st.text_input("Şifre", type="password")
            submitted = st.form_submit_button("Giriş yap", type="primary", use_container_width=True)

        if submitted:
            users = load_users()
            if username in users and users[username]["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.session_state["role"] = users[username].get("role", "user")
                write_log("LOGIN", "Kullanıcı giriş yaptı")
                st.rerun()
            else:
                st.error("Kullanıcı adı veya şifre hatalı.")


def admin_panel():
    users = load_users()

    with st.sidebar.expander("Kullanıcı yönetimi", expanded=False):
        st.markdown("**Yeni kullanıcı ekle**")
        new_user = st.text_input("Yeni kullanıcı adı", key="new_user")
        new_pass = st.text_input("Yeni şifre", type="password", key="new_pass")
        new_role = st.selectbox("Rol", ["user", "admin"], key="new_role")
        if st.button("Kullanıcı ekle", key="add_user_btn"):
            if not new_user.strip() or not new_pass.strip():
                st.warning("Kullanıcı adı ve şifre boş olamaz.")
            elif new_user in users:
                st.warning("Bu kullanıcı zaten mevcut.")
            else:
                users[new_user] = {"password": new_pass, "role": new_role}
                save_users(users)
                write_log("ADD_USER", f"Yeni kullanıcı eklendi: {new_user} ({new_role})")
                st.success("Kullanıcı eklendi.")
                st.rerun()

        st.markdown("---")
        st.markdown("**Şifre değiştir**")
        selected_user = st.selectbox("Kullanıcı", list(users.keys()), key="selected_user")
        changed_password = st.text_input("Yeni şifre", type="password", key="changed_password")
        if st.button("Şifreyi güncelle", key="change_pass_btn"):
            if not changed_password.strip():
                st.warning("Yeni şifre boş olamaz.")
            else:
                users[selected_user]["password"] = changed_password
                save_users(users)
                write_log("CHANGE_PASSWORD", f"Şifre güncellendi: {selected_user}")
                st.success("Şifre güncellendi.")
                st.rerun()

        st.markdown("---")
        st.markdown("**Kullanıcı sil**")
        deletable = [u for u in users if u != "admin"]
        if deletable:
            delete_user = st.selectbox("Silinecek kullanıcı", deletable, key="delete_user")
            if st.button("Kullanıcıyı sil", key="delete_user_btn"):
                users.pop(delete_user, None)
                save_users(users)
                write_log("DELETE_USER", f"Kullanıcı silindi: {delete_user}")
                st.success("Kullanıcı silindi.")
                st.rerun()
        else:
            st.caption("Silinebilir kullanıcı yok.")

    with st.sidebar.expander("Kullanıcı işlem kayıtları", expanded=False):
        logs_df = read_logs()
        if logs_df.empty:
            st.caption("Henüz kayıt yok.")
        else:
            st.dataframe(logs_df.iloc[::-1], width="stretch", height=260)
            st.download_button(
                "Log dosyasını indir",
                data=logs_df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="hey_roi_activity_log.csv",
                mime="text/csv"
            )


# -----------------------------
# Tesseract setup
# -----------------------------
def configure_tesseract() -> str:
    candidates = [
        shutil.which("tesseract"),
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for c in candidates:
        if c and Path(c).exists():
            pytesseract.pytesseract.tesseract_cmd = c
            return c
    return ""


TESSERACT_PATH = configure_tesseract()


# -----------------------------
# Clinical blocks
# -----------------------------
BLOCKS = {
    "A1": """Hemoglobin elektroforezinde HbF, HbA ve HbA2 fraksiyonları değerlendirilmiştir. Bu yaş grubunda hemoglobin fraksiyonları fizyolojik maturasyon sürecinde olduğundan, HbF düzeyinin yüksek izlenmesi ve HbA2 değerlerinin tanısal olmaması beklenebilir. Mevcut bulgular yaşa uygun hemoglobin dağılımı ile uyumludur.

Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.""",

    "A2": """HbF düzeyinin yaş grubuna göre üst sınıra yakın olduğu izlenmiştir. Bu yaş döneminde HbF yüksekliği fizyolojik olarak görülebilmekle birlikte, klinik ve hematolojik bulgular ile birlikte değerlendirilmesi ve izlem önerilir.

Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.""",

    "A3": """Hemoglobin elektroforezinde HbF düzeyinin yüksek olduğu izlenmiştir. 0–12 ay döneminde HbF yüksekliği fizyolojik olmakla birlikte, yaşa göre belirgin yüksek değerlerde dikkatli değerlendirme gereklidir.

Bu kapsamda öncelikle hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte değerlendirilmesi önerilir.
HbA düzeyinin düşük bulunması veya klinik olarak anlamlı anemi varlığında; β-talasemi major/intermedia, herediter fetal hemoglobin persistansı (HPFH) veya δβ-talasemi ayırıcı tanıda düşünülmelidir.
Klinik gereklilik halinde moleküler genetik inceleme (HPFH/δβ-talasemi delesyon analizi; HBB, HBD ve HBG genlerini kapsayan analiz) planlanabilir.
Kesin değerlendirme için hemoglobin elektroforezinin 3–4 yaş arasında tekrar edilmesi önerilir.""",

    "B1": """Hemoglobin elektroforezinde yaşa uygun hemoglobin dağılımı izlenmiştir. Yapısal hemoglobin varyantı lehine ek bir pik saptanmamıştır.

Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir. HbA2 normal olsa dahi, hemogramda mikrositoz veya klinik şüphe varlığında hemoglobinopati açısından ileri değerlendirme önerilir.""",

    "B2": """HbA2 düzeyinin sınırda yüksek olduğu izlenmiştir. Bu bulgu tek başına tanısal olmayıp β-talasemi taşıyıcılığı açısından değerlendirilmelidir.
Bu yaş grubunda kesin tanı için ileri yaşta tekrar değerlendirme önerilir.
Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.""",

    "B3": """HbA2 düzeyinin yüksek olduğu izlenmiştir. β-talasemi taşıyıcılığı açısından değerlendirilmelidir. Bu yaş grubunda kesin tanı için ileri yaşta (≥6 yaş) tekrar değerlendirme önerilir.
Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.""",

    "B4": """HbF düzeyinin yaşa göre referans aralığının üzerinde olduğu izlenmiştir. Bu bulgu herediter fetal hemoglobin persistansı (HPFH) veya δβ-talasemi ile uyumlu olabilir.
Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.
Klinik gereklilik halinde β-globin gen kümesine yönelik moleküler genetik inceleme planlanabilir.""",

    "C1": """Hemoglobin elektroforezinde HbA, HbA2 ve HbF dağılımı yaşa uygun referans aralıkları ile uyumludur. Yapısal hemoglobin varyantı lehine ek bir pik saptanmamıştır.

Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir. HbA2 normal olsa dahi, hemogramda mikrositoz veya klinik şüphe varlığında hemoglobinopati açısından ileri değerlendirme önerilir.""",

    "C2": """HbA2 düzeyinin sınırda yüksek olduğu izlenmiştir. Bu bulgu tek başına tanısal olmayıp β-talasemi taşıyıcılığı açısından değerlendirilmelidir.
Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.""",

    "C3": """HbA2 düzeyinin yüksek olduğu izlenmiştir. Bu bulgu β-talasemi taşıyıcılığı ile uyumlu olabilir.
Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.""",

    "C4": """Hemoglobin elektroforezinde HbA2 düzeyinin referans aralığının üzerinde olduğu izlenmiştir. HbF düzeyi normal/sınırdadır. Bu bulgular β-talasemi taşıyıcılığı (β-talasemi minör) ile uyumludur. Yapısal hemoglobin varyantı lehine ek bir pik saptanmamıştır.
Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.
18 yaş üstü hastalarda: Aile planlaması açısından eş/partnerde hemoglobinopati taşıyıcılığı yönünden değerlendirme (Hb elektroforezi ± hemogram) önerilir. Klinik gereklilik halinde β-globin gen analizi (HBB) yapılabilir.""",

    "C5": """HbF düzeyinin yaşa göre referans aralığının üzerinde olduğu izlenmiştir. HbA2 düzeyi normal/sınırdadır. Bu hemoglobin dağılımı herediter fetal hemoglobin persistansı (HPFH) veya δβ-talasemi ile uyumlu olabilir.
Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.
Klinik gereklilik halinde β-globin gen kümesine yönelik moleküler genetik inceleme planlanabilir.""",

    "SPECIAL": "HbF belirgin yüksek ve HbA yok/düşük izlenmektedir. Bu patern β-talasemi major ile uyumlu olabilir. Sonuç sorumlu hekime iletilmelidir.",
}
LOW_HBA_THRESHOLD = 10.0


def classify_age_group(age_years: float) -> str:
    if age_years < 1:
        return "A"
    elif age_years < 6:
        return "B"
    return "C"


def generate_comment(age_years: float, sex: str, hba2: float, hbf: float, hba: float, extra_variant_peak: bool):
    age_group = classify_age_group(age_years)

    if extra_variant_peak:
        return {
            "block": None,
            "comment": "Yapısal hemoglobin varyantı lehine ek pik/varyant bildirildiği için bu standart algoritma tek başına uygulanmamalıdır. Sonuç sorumlu hekime yönlendirilmelidir.",
            "reason": "Algoritma blokları yapısal varyant saptanmayan standart dağılımlar için tanımlanmıştır."
        }

    if hbf >= 90 and hba <= LOW_HBA_THRESHOLD:
        return {
            "block": "SPECIAL",
            "comment": BLOCKS["SPECIAL"],
            "reason": "HbF %90 ve üzeri, HbA yok/düşük özel durumu"
        }

    if age_group == "A":
        if 5 <= hbf <= 20:
            return {"block": "A1", "comment": BLOCKS["A1"], "reason": "0–12 ay, HbF %5–20"}
        elif 20 < hbf <= 25:
            return {"block": "A2", "comment": BLOCKS["A2"], "reason": "0–12 ay, HbF %20–25"}
        elif hbf > 25:
            return {"block": "A3", "comment": BLOCKS["A3"], "reason": "0–12 ay, HbF >%25"}
        return {"block": None, "comment": "Girilen HbF değeri A bloklarında tanımlı aralıklarla eşleşmiyor. Sonuç sorumlu hekime yönlendirilmelidir.", "reason": "Tanımsız aralık"}

    if age_group == "B":
        if hba2 >= 4.5:
            return {"block": "B3", "comment": BLOCKS["B3"], "reason": "1–6 yaş, HbA2 ≥4.5"}
        elif 3.5 <= hba2 < 4.5:
            return {"block": "B2", "comment": BLOCKS["B2"], "reason": "1–6 yaş, HbA2 3.5–4.5"}
        elif hba2 < 3.5:
            if hbf > 5:
                return {"block": "B4", "comment": BLOCKS["B4"], "reason": "1–6 yaş, HbA2 <3.5 ve HbF >5"}
            return {"block": "B1", "comment": BLOCKS["B1"], "reason": "1–6 yaş, HbA2 <3.5"}

    if age_group == "C":
        if hba2 >= 5.0:
            return {"block": "C4", "comment": BLOCKS["C4"], "reason": "≥6 yaş, HbA2 ≥5.0"}
        elif 4.0 <= hba2 < 5.0:
            return {"block": "C3", "comment": BLOCKS["C3"], "reason": "≥6 yaş, HbA2 4.0–5.0"}
        elif 3.5 <= hba2 < 4.0:
            if hbf > 5:
                return {"block": "C5", "comment": BLOCKS["C5"], "reason": "≥6 yaş, HbF >5 ve HbA2 sınırda"}
            return {"block": "C2", "comment": BLOCKS["C2"], "reason": "≥6 yaş, HbA2 3.5–4.0"}
        elif hba2 < 3.5:
            if hbf > 5:
                return {"block": "C5", "comment": BLOCKS["C5"], "reason": "≥6 yaş, HbA2 normal ve HbF >5"}
            return {"block": "C1", "comment": BLOCKS["C1"], "reason": "≥6 yaş, HbA2 <3.5"}

    return {
        "block": None,
        "comment": "Algoritma ile eşleşmeyen bir durum oluştu. Sonuç sorumlu hekime yönlendirilmelidir.",
        "reason": "Belirsiz durum"
    }


def pdf_first_page_to_image(file_bytes: bytes):
    if PDF2IMAGE_OK:
        try:
            imgs = convert_from_bytes(file_bytes, dpi=250, first_page=1, last_page=1)
            if imgs:
                return imgs[0]
        except Exception:
            pass
    if PYMUPDF_OK:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
            return img
        except Exception:
            pass
    return None


def load_image(uploaded_file):
    suffix = uploaded_file.name.lower().split(".")[-1]
    file_bytes = uploaded_file.read()
    if suffix == "pdf":
        return pdf_first_page_to_image(file_bytes), file_bytes
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB"), file_bytes
    except Exception:
        return None, file_bytes


def preprocess_variants(img: Image.Image):
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    sharp = gray.filter(ImageFilter.SHARPEN)
    sharp = ImageEnhance.Contrast(sharp).enhance(1.2)

    arr = np.array(sharp)
    bin170 = Image.fromarray(np.where(arr > 170, 255, 0).astype(np.uint8))
    bin185 = Image.fromarray(np.where(arr > 185, 255, 0).astype(np.uint8))
    bin200 = Image.fromarray(np.where(arr > 200, 255, 0).astype(np.uint8))

    enlarged = sharp.resize((sharp.width * 2, sharp.height * 2))
    padded = ImageOps.expand(enlarged, border=30, fill="white")

    return {
        "gray": gray,
        "sharp": sharp,
        "binary170": bin170,
        "binary185": bin185,
        "binary200": bin200,
        "enlarged": enlarged,
        "padded": padded,
    }


def ocr_text(img: Image.Image, psm=6) -> str:
    cfg = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(img, lang="eng", config=cfg)


def normalize_peak_name(text: str) -> str:
    t = text.strip()
    t = re.sub(r"[|]", " ", t)
    t = re.sub(r"[^A-Za-z0-9\-/\. ]", "", t)
    t = t.replace("AO", "A0").replace("A O", "A0").replace("A@", "A0")
    t = t.replace("AZ", "A2").replace("A Z", "A2")
    t = t.replace("S Window", "S-Window").replace("SWindow", "S-Window")
    t = t.replace("Unknownn", "Unknown")
    t = re.sub(r"\bAla\b", "A1a", t)
    t = re.sub(r"\bAlb\b", "A1b", t)
    t = t.replace("LAIc/CHb-1", "LA1c/CHb-1").replace("LAlc/CHb-1", "LA1c/CHb-1")
    return t.strip()


def parse_percent_value(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def parse_rows_from_text(text: str) -> pd.DataFrame:
    rows = []
    for raw in text.splitlines():
        line = raw.strip().replace(",", ".")
        if not line:
            continue
        if "peak table" in line.lower() or "total area" in line.lower() or "concentration" in line.lower():
            continue

        m = re.search(
            r"(?P<peak>[A-Za-z0-9\-/\. ]+?)\s+"
            r"(?P<rtime>\d+(?:\.\d+)?)\s+"
            r"(?P<height>\d+)\s+"
            r"(?P<area>\d+)\s+"
            r"(?P<area_pct>[<>]?\s*\d+(?:\.\d+)?)$",
            line
        )
        if m:
            rows.append({
                "Peak": normalize_peak_name(m.group("peak")),
                "R.time": m.group("rtime"),
                "Height": m.group("height"),
                "Area": m.group("area"),
                "Area %": m.group("area_pct"),
            })
    return pd.DataFrame(rows)


def parse_rows_by_roi_columns(roi_img: Image.Image):
    w, h = roi_img.size
    peak_img = roi_img.crop((0, 0, int(w * 0.48), h))
    area_pct_img = roi_img.crop((int(w * 0.74), 0, w, h))

    best_peak_text = ""
    best_pct_text = ""
    best_peak_score = -1
    best_pct_score = -1

    for _, variant in preprocess_variants(peak_img).items():
        txt = ocr_text(variant, psm=6)
        candidates = [normalize_peak_name(x) for x in txt.splitlines() if x.strip()]
        score = sum(1 for x in candidates if re.search(r"[A-Za-z]", x))
        if score > best_peak_score:
            best_peak_score = score
            best_peak_text = txt

    for _, variant in preprocess_variants(area_pct_img).items():
        txt = ocr_text(variant, psm=6)
        cands = [x.strip().replace(",", ".") for x in txt.splitlines() if x.strip()]
        score = sum(1 for x in cands if re.search(r"^[<>]?\s*\d+(?:\.\d+)?$", x))
        if score > best_pct_score:
            best_pct_score = score
            best_pct_text = txt

    peaks = []
    for line in best_peak_text.splitlines():
        n = normalize_peak_name(line)
        if not n:
            continue
        if n.lower() in {"peak", "r.time", "height", "area", "area %", "area%"}:
            continue
        if re.search(r"^\d", n):
            continue
        peaks.append(n)

    pcts = []
    for line in best_pct_text.splitlines():
        line = line.strip().replace(",", ".")
        if re.search(r"^[<>]?\s*\d+(?:\.\d+)?$", line):
            pcts.append(line)

    max_len = max(len(peaks), len(pcts), 1)
    peaks += [""] * (max_len - len(peaks))
    pcts += [""] * (max_len - len(pcts))

    return pd.DataFrame({
        "Peak": peaks,
        "R.time": [""] * max_len,
        "Height": [""] * max_len,
        "Area": [""] * max_len,
        "Area %": pcts,
    })


def parse_concentration_box(roi_img: Image.Image):
    w, h = roi_img.size
    crop = roi_img.crop((0, int(h * 0.72), int(w * 0.5), h))
    best_text = ""
    best_score = -1

    for _, variant in preprocess_variants(crop).items():
        txt = ocr_text(variant, psm=6)
        score = 0
        if re.search(r"%\s*A2", txt, flags=re.I):
            score += 2
        if re.search(r"\bA2\b", txt, flags=re.I):
            score += 1
        if re.search(r"\d+(?:\.\d+)?", txt):
            score += 1
        if score > best_score:
            best_score = score
            best_text = txt

    result = {}
    for line in best_text.splitlines():
        t = line.strip().replace(",", ".")
        if re.search(r"%\s*A2", t, flags=re.I):
            val = parse_percent_value(t)
            if val is not None:
                result["A2"] = val
        if re.search(r"%\s*A1c", t, flags=re.I):
            val = parse_percent_value(t)
            if val is not None:
                result["A1c"] = val
    return result


def merge_candidate_tables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df1 is not None and not df1.empty:
        return df1
    if df2 is not None and not df2.empty:
        return df2
    return pd.DataFrame(columns=["Peak", "R.time", "Height", "Area", "Area %"])


def deduplicate_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    temp = df.copy()
    temp["Peak_norm"] = temp["Peak"].astype(str).apply(normalize_peak_name)

    rows = []
    for peak_name, grp in temp.groupby("Peak_norm", sort=False):
        if peak_name == "Unknown":
            for _, row in grp.iterrows():
                row_dict = row.drop(labels=["Peak_norm"]).to_dict()
                row_dict["Peak"] = "Unknown"
                rows.append(row_dict)
            continue

        base = grp.iloc[0].drop(labels=["Peak_norm"]).to_dict()
        base["Peak"] = peak_name
        for col in ["R.time", "Height", "Area", "Area %"]:
            if (not str(base.get(col, "")).strip()) and grp[col].astype(str).str.strip().ne("").any():
                first_nonempty = grp.loc[grp[col].astype(str).str.strip().ne(""), col].iloc[0]
                base[col] = first_nonempty
        rows.append(base)

    return pd.DataFrame(rows)


def ensure_default_peak_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({
            "Peak": DEFAULT_PEAK_ORDER,
            "R.time": "",
            "Height": "",
            "Area": "",
            "Area %": "",
        })

    df = df.copy()
    existing = [str(x).strip().lower() for x in df["Peak"].fillna("").tolist()]
    rows_to_add = []
    for peak in DEFAULT_PEAK_ORDER:
        if peak.lower() not in existing:
            rows_to_add.append({"Peak": peak, "R.time": "", "Height": "", "Area": "", "Area %": ""})

    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

    order_map = {peak.lower(): i for i, peak in enumerate(DEFAULT_PEAK_ORDER)}
    df["_sort"] = df["Peak"].astype(str).str.lower().map(lambda x: order_map.get(x, len(DEFAULT_PEAK_ORDER) + 5))
    df = df.sort_values(["_sort"], kind="stable").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def apply_concentration_fallback(df: pd.DataFrame, conc_dict: dict | None):
    if df is None or df.empty or not conc_dict:
        return df
    out = df.copy()

    if "A2" in conc_dict:
        mask = out["Peak"].astype(str).str.lower().eq("a2")
        if mask.any():
            current = out.loc[mask, "Area %"].iloc[0]
            if parse_percent_value(current) is None:
                out.loc[mask, "Area %"] = str(conc_dict["A2"])

    if "A1c" in conc_dict:
        mask = out["Peak"].astype(str).str.lower().eq("a1c")
        if mask.any():
            current = out.loc[mask, "Area %"].iloc[0]
            if parse_percent_value(current) is None:
                out.loc[mask, "Area %"] = str(conc_dict["A1c"])

    return out


def standardize_table(df: pd.DataFrame, concentration_fallback: dict | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        out = ensure_default_peak_rows(pd.DataFrame(columns=["Peak", "R.time", "Height", "Area", "Area %"]))
        return apply_concentration_fallback(out, concentration_fallback)

    out = df.copy()
    for col in ["Peak", "R.time", "Height", "Area", "Area %"]:
        if col not in out.columns:
            out[col] = ""
    out = out[["Peak", "R.time", "Height", "Area", "Area %"]]
    out["Peak"] = out["Peak"].astype(str).apply(normalize_peak_name)
    out = out[~out["Peak"].astype(str).str.lower().isin(["c-window", "d-window"])].copy()
    out["Area %"] = out["Area %"].astype(str).str.replace(",", ".", regex=False)
    out = deduplicate_table(out)
    out = ensure_default_peak_rows(out)
    out = apply_concentration_fallback(out, concentration_fallback)
    return out


def extract_named_peaks(df: pd.DataFrame) -> dict:
    result = {
        "HbA2": None,
        "HbF": None,
        "S-Window": None,
        "Unknown": None,
    }
    for _, row in df.iterrows():
        peak = str(row.get("Peak", "")).strip().lower()
        pct = parse_percent_value(row.get("Area %", ""))
        if pct is None:
            continue

        if peak == "a2":
            result["HbA2"] = pct
        elif peak in {"f", "hbf", "hb f"} or "f-window" in peak:
            result["HbF"] = pct
        elif "s-window" in peak or peak == "s":
            result["S-Window"] = pct
        elif "unknown" in peak:
            result["Unknown"] = pct if result["Unknown"] is None else max(result["Unknown"], pct)
    return result


def calculate_hba_from_table(df: pd.DataFrame, s_threshold=5.0, unknown_threshold=5.0):
    named = extract_named_peaks(df)

    hba2 = named["HbA2"] or 0.0
    hbf = named["HbF"] or 0.0

    s_val = named["S-Window"] if named["S-Window"] is not None and named["S-Window"] > s_threshold else 0.0
    u_val = named["Unknown"] if named["Unknown"] is not None and named["Unknown"] > unknown_threshold else 0.0

    total_excluded = hba2 + hbf + s_val + u_val
    calc_hba = round(max(0.0, 100.0 - total_excluded), 1)

    included = [("HbA (100 - diğer fraksiyonlar)", calc_hba)]
    excluded = [
        ("HbA2", hba2, "100'den düşülen"),
        ("HbF", hbf, "100'den düşülen"),
    ]
    if s_val > 0:
        excluded.append(("S-Window", s_val, f"100'den düşülen (> %{s_threshold})"))
    if u_val > 0:
        excluded.append(("Unknown", u_val, f"100'den düşülen (> %{unknown_threshold})"))

    return calc_hba, included, excluded


def variant_flag(named_peaks: dict, threshold=5.0):
    reasons = []
    s = named_peaks.get("S-Window")
    u = named_peaks.get("Unknown")
    if s is not None and s > threshold:
        reasons.append(f"S-Window %{s:.1f} > %{threshold}")
    if u is not None and u > threshold:
        reasons.append(f"Unknown %{u:.1f} > %{threshold}")
    return len(reasons) > 0, reasons


def build_variant_comment(named_peaks: dict, repeat_confirmed: bool):
    phrases = []
    s = named_peaks.get("S-Window")
    u = named_peaks.get("Unknown")

    if s is not None:
        phrases.append(f"S penceresinde (~%{s:.1f})")
    if u is not None:
        phrases.append(f"farklı bir bölgede anlamlı düzeyde ek bir hemoglobin fraksiyonu (~%{u:.1f})")

    if phrases:
        first = "Hemoglobin elektroforezinde normal hemoglobin fraksiyonlarına ek olarak " + " ve ".join(phrases) + " izlenmiştir."
    else:
        first = "Hemoglobin elektroforezinde yapısal hemoglobin varyantı lehine ek pik izlenmiştir."

    parts = [first]
    if repeat_confirmed:
        parts.append("Bulgular tekrar çalışmada benzer şekilde saptanmış olup analitik hata olasılığı dışlanmıştır.")
    parts.append("Bu bulgular hemoglobin varyantı varlığını düşündürmektedir.")
    parts.append("Mevcut elektroforetik/HPLC yöntem ile varyantın tipi kesin olarak ayırt edilemeyebilir.")
    parts.append("Sonuçların değerlendirilmesinde klinik bulgular, hemogram (özellikle Hb, MCV, MCH) ve demir metabolizması parametreleri (ferritin, serum demir, TDBK/TSAT) ile birlikte yorumlanması önerilir.")
    parts.append("Varyantın karakterizasyonu için kapiller elektroforez ile doğrulama ve gerekli durumlarda moleküler inceleme (β-globin gen analizi; HBB gen sekans analizi) önerilir.")
    return "\n\n".join(parts)


def df_to_text(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    lines = []
    for _, row in df.iterrows():
        vals = [str(row.get(c, "")).strip() for c in ["Peak", "R.time", "Height", "Area", "Area %"]]
        if any(vals):
            lines.append("\t".join(vals))
    return "\n".join(lines)


def fmt_optional_pct(value):
    return "-" if value is None else f"{value:.1f}"


def prepare_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Peak", "R.time", "Height", "Area", "Area %"]
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str).str.strip()
    return out[cols]


def compare_peak_tables(original_df: pd.DataFrame, edited_df: pd.DataFrame):
    original = prepare_comparison_df(original_df)
    edited = prepare_comparison_df(edited_df)

    def with_occ(df):
        temp = df.copy()
        temp["_peak_norm"] = temp["Peak"].apply(normalize_peak_name)
        temp["_occ"] = temp.groupby("_peak_norm").cumcount() + 1
        temp["_key"] = temp["_peak_norm"] + "#" + temp["_occ"].astype(str)
        return temp

    o = with_occ(original)
    e = with_occ(edited)

    all_keys = sorted(set(o["_key"]).union(set(e["_key"])))
    changes = []
    changed_cells = set()

    for key in all_keys:
        o_row = o[o["_key"] == key]
        e_row = e[e["_key"] == key]

        if o_row.empty and not e_row.empty:
            row = e_row.iloc[0]
            changes.append(f"Yeni satır eklendi: Peak={row['Peak']}, R.time={row['R.time']}, Height={row['Height']}, Area={row['Area']}, Area %={row['Area %']}")
            continue

        if e_row.empty and not o_row.empty:
            row = o_row.iloc[0]
            changes.append(f"Satır silindi: Peak={row['Peak']}, R.time={row['R.time']}, Height={row['Height']}, Area={row['Area']}, Area %={row['Area %']}")
            continue

        o_row = o_row.iloc[0]
        e_row = e_row.iloc[0]
        peak_label = e_row["Peak"] or o_row["Peak"] or key

        field_changes = []
        for field in ["Peak", "R.time", "Height", "Area", "Area %"]:
            old = str(o_row[field]).strip()
            new = str(e_row[field]).strip()
            if old != new:
                field_changes.append(f"{field}: '{old}' -> '{new}'")
                changed_cells.add((key, field))

        if field_changes:
            changes.append(f"{peak_label}: " + "; ".join(field_changes))

    return changes, changed_cells


def build_display_df_with_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_comparison_df(df)
    out["_peak_norm"] = out["Peak"].apply(normalize_peak_name)
    out["_occ"] = out.groupby("_peak_norm").cumcount() + 1
    out["_key"] = out["_peak_norm"] + "#" + out["_occ"].astype(str)
    return out


# -----------------------------
# Main app
# -----------------------------
ensure_log_file()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

st.title("Hemoglobin Elektroforez Yorum Destek Sistemi")
st.caption(f"{APP_NAME} {APP_VERSION} • ROI tabanlı Peak Table okuma + HbA hesaplama + varyant kararı")
st.caption(AUTHOR_LINE)

with st.sidebar:
    st.write(f"Giriş yapan kullanıcı: **{st.session_state.get('username', '')}**")
    st.write(f"Rol: **{st.session_state.get('role', '')}**")
    if st.button("Çıkış yap"):
        write_log("LOGOUT", "Kullanıcı çıkış yaptı")
        st.session_state.clear()
        st.rerun()

if st.session_state.get("role") == "admin":
    admin_panel()

with st.sidebar:
    st.markdown("---")

    st.subheader("Kurallar")
    threshold = st.number_input("S-Window / Unknown varyant eşiği (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    repeat_confirmed = st.checkbox("Tekrar çalışmada benzer bulundu / analitik hata dışlandı", value=True)
    st.caption("Bu kutuyu, aynı örneğin tekrar analizinde benzer pik dağılımı saptandıysa ve bulgunun cihaz/analitik hatadan kaynaklanmadığını düşünüyorsanız işaretleyin.")

    st.markdown("""
**Uygulama notları**
- Peak Table alanını ROI ile seçmeniz beklenir.
- Varyant lehine bulgu kuralı:
  - S-Window > %5
  - Unknown > %5
- HbA hesaplama kuralı:
  - HbA = 100 - (HbA2 + HbF + S-Window(>%eşik) + Unknown(>%eşik))
- OCR hatası olasılığı nedeniyle tabloyu gözle doğrulayıp düzeltmeniz önerilir.
""")

    st.markdown("---")
    st.subheader("Sistem durumu")
    st.write(f"Tesseract: {'Bulundu' if TESSERACT_PATH else 'Bulunamadı'}")
    if TESSERACT_PATH:
        st.code(TESSERACT_PATH)
    st.write(f"pdf2image: {'Hazır' if PDF2IMAGE_OK else 'Yok'}")
    st.write(f"PyMuPDF: {'Hazır' if PYMUPDF_OK else 'Yok'}")
    st.write(f"streamlit-cropper: {'Hazır' if CROPPER_OK else 'Yok'}")

uploaded = st.file_uploader("PDF veya görüntü yükleyin", type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"])


base_img = None
uploaded_name = ""
if uploaded is not None:
    uploaded_name = uploaded.name
    base_img, _ = load_image(uploaded)

if base_img is None:
    st.info("Başlamak için PDF veya görüntü yükleyin.")
    st.stop()

if st.session_state.get("last_logged_upload") != uploaded_name and uploaded_name:
    write_log("UPLOAD_FILE", f"Dosya yüklendi: {uploaded_name}")
    st.session_state["last_logged_upload"] = uploaded_name

st.subheader("Yüklenen sayfa / görüntü")
st.image(base_img, caption="Kaynak görsel", width="stretch")


st.subheader("Peak Table ROI seçimi")


if CROPPER_OK:
    st.info("Fare ile Peak Table alanını seçin. A2 satırı ve mümkünse alt konsantrasyon kutusu da seçim alanına dahil olsun.")
    roi_img = st_cropper(
        base_img,
        realtime_update=False,
        box_color="#4F8BF9",
        aspect_ratio=None,
        return_type="image"
    )
else:
    st.warning("İnteraktif cropper devre dışı. Manuel ROI seçimi form ile kullanılıyor.")

    w, h = base_img.size
    default_left = int(w * 0.08)
    default_top = int(h * 0.55)
    default_right = int(w * 0.82)
    default_bottom = int(h * 0.95)

    if "roi_left" not in st.session_state:
        st.session_state.roi_left = default_left
    if "roi_top" not in st.session_state:
        st.session_state.roi_top = default_top
    if "roi_right" not in st.session_state:
        st.session_state.roi_right = default_right
    if "roi_bottom" not in st.session_state:
        st.session_state.roi_bottom = default_bottom
    if "roi_ready" not in st.session_state:
        st.session_state.roi_ready = False


    with st.form("manual_roi_form"):
        
        
        c1, c2 = st.columns(2)
        with c1:
            left = st.slider("Sol", 0, w - 1, st.session_state.roi_left)
            top = st.slider("Üst", 0, h - 1, st.session_state.roi_top)
        with c2:
            right = st.slider("Sağ", 1, w, st.session_state.roi_right)
            bottom = st.slider("Alt", 1, h, st.session_state.roi_bottom)

        apply_roi = st.form_submit_button("ROI'yi uygula", type="primary")


    if apply_roi:
        st.session_state.roi_left = left
        st.session_state.roi_top = top
        st.session_state.roi_right = right
        st.session_state.roi_bottom = bottom
        st.session_state.roi_ready = True
        st.session_state.ocr_ready = False

    if not st.session_state.get("roi_ready"):
        st.stop()

    if st.session_state.roi_right <= st.session_state.roi_left:
        st.session_state.roi_right = min(w, st.session_state.roi_left + 10)
    if st.session_state.roi_bottom <= st.session_state.roi_top:
        st.session_state.roi_bottom = min(h, st.session_state.roi_top + 10)

    roi_img = base_img.crop((
        st.session_state.roi_left,
        st.session_state.roi_top,
        st.session_state.roi_right,
        st.session_state.roi_bottom
    ))

    if not st.session_state.roi_ready:
        st.info("Önce ROI alanını ayarlayıp 'ROI'yi uygula' butonuna basın.")
        st.stop()

st.image(roi_img, caption="Seçilen Peak Table ROI", width="stretch")

if "ocr_ready" not in st.session_state:
    st.session_state.ocr_ready = False

if st.session_state.parsed_df is None:
    if st.button("ROI sonrası OCR / tablo okuma işlemini başlat", type="primary"):
        concentration_fallback = parse_concentration_box(roi_img)
        parsed_direct = standardize_table(parse_rows_from_text(best_text), concentration_fallback=concentration_fallback)
        parsed_roi_columns = standardize_table(parse_rows_by_roi_columns(roi_img), concentration_fallback=concentration_fallback)
        parsed_df = standardize_table(
            merge_candidate_tables(parsed_direct, parsed_roi_columns),
            concentration_fallback=concentration_fallback
        )

        st.session_state.concentration_fallback = concentration_fallback
        st.session_state.parsed_df = parsed_df
        st.rerun()
    else:
        st.info("Önce ROI'yi uygula, sonra OCR / tablo okuma işlemini bu butonla başlat.")
    st.stop()

if "parsed_df" not in st.session_state:
    st.session_state.parsed_df = None

if "concentration_fallback" not in st.session_state:
    st.session_state.concentration_fallback = None


st.subheader("ROI OCR ve Peak Table ayrıştırma")

best_text = ""
best_score = -1
variant_images = preprocess_variants(roi_img)
for _, img_variant in variant_images.items():
    txt = ocr_text(img_variant, psm=6)
    df_candidate = parse_rows_from_text(txt)
    score = len(df_candidate)
    if score > best_score:
        best_score = score
        best_text = txt

concentration_fallback = st.session_state.concentration_fallback
parsed_df = st.session_state.parsed_df

st.dataframe(parsed_df, width="stretch")

st.subheader("Peak Table doğrulama / manuel düzeltme")
edited_df = st.data_editor(
    parsed_df,
    num_rows="dynamic",
    width="stretch",
    key="editor"
)

manual_text = st.text_area(
    "İsterseniz Peak Table satırlarını manuel yapıştırın",
    height=120,
    placeholder="A0\t1.75\t119886\t660888\t54.5\nA2\t3.27\t1097\t22668\t2.3\nUnknown\t3.99\t47348\t296785\t24.5"
)
if manual_text.strip():
    rows = []
    for line in manual_text.splitlines():
        parts = re.split(r"[\t;]+", line.strip())
        if len(parts) >= 5:
            rows.append({
                "Peak": parts[0],
                "R.time": parts[1],
                "Height": parts[2],
                "Area": parts[3],
                "Area %": parts[4],
            })
        elif len(parts) >= 2:
            rows.append({
                "Peak": parts[0],
                "R.time": "",
                "Height": "",
                "Area": "",
                "Area %": parts[-1],
            })
    if rows:
        edited_df = standardize_table(pd.DataFrame(rows), concentration_fallback=concentration_fallback)
        st.info("Manuel yapıştırılan metin uygulandı.")
        st.dataframe(edited_df, width="stretch")

edited_df = standardize_table(pd.DataFrame(edited_df), concentration_fallback=concentration_fallback)

# changed-cell highlighting preview
changes_preview, changed_cells_preview = compare_peak_tables(parsed_df, edited_df)
display_df = build_display_df_with_keys(edited_df)

def highlight_changed(row):
    styles = []
    row_key = row["_key"]
    for col in ["Peak", "R.time", "Height", "Area", "Area %"]:
        if (row_key, col) in changed_cells_preview:
            styles.append("background-color: #fff3b0; color: #7a4b00; font-weight: 600;")
        else:
            styles.append("")
    styles.extend(["", "", ""])  # hidden helper cols safety
    return styles

st.markdown("**Değiştirilen hücreler renkli önizleme**")
styled = display_df.style.apply(highlight_changed, axis=1)
st.dataframe(
    styled.hide(axis="columns", subset=["_peak_norm", "_occ", "_key"]),
    width="stretch"
)

st.subheader("Hesaplanan ana parametreler")

named = extract_named_peaks(edited_df)
calc_hba, included_peaks, excluded_peaks = calculate_hba_from_table(
    edited_df,
    s_threshold=threshold,
    unknown_threshold=threshold
)
is_variant, variant_reasons = variant_flag(named, threshold=threshold)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("HbA (hesaplanan)", f"{calc_hba:.1f}%")
with col2:
    st.metric("HbA2", fmt_optional_pct(named["HbA2"]))
with col3:
    st.metric("HbF", fmt_optional_pct(named["HbF"]))
with col4:
    st.metric("Varyant lehine bulgu", "Evet" if is_variant else "Hayır")

st.write("**Varyant gerekçesi:** " + (", ".join(variant_reasons) if variant_reasons else "Yok"))

with st.expander("HbA hesaplama detayları"):
    st.write("**HbA hesaplanan formül**")
    st.write("HbA = 100 - (HbA2 + HbF + S-Window(>%eşik) + Unknown(>%eşik))")
    st.write("**Hesap bileşenleri**")
    st.dataframe(pd.DataFrame(excluded_peaks, columns=["Bileşen", "Değer", "Açıklama"]), width="stretch")


st.subheader("Klinik girişler")
c1, c2 = st.columns(2)
with c1:
    age_years = st.number_input("Yaş (yıl)", min_value=0.0, max_value=120.0, value=18.0, step=0.1)
with c2:
    sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek", "Belirtilmedi"])
    

manual_override_variant = st.checkbox("Varyant lehine bulgu olarak manuel işaretle", value=False)


if st.button("Klinik yorumu üret", type="primary"):
    st.write("OK 6")
    st.stop()

    
