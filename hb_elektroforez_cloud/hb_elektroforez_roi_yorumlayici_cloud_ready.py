
import io
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# Optional deps
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_OK = True
except Exception:
    PDF2IMAGE_OK = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except Exception:
    PYMUPDF_OK = False

try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except Exception:
    CROPPER_OK = False


st.set_page_config(
    page_title="Hb Elektroforez ROI Yorumlayıcı",
    layout="wide"
)

st.title("Hemoglobin Elektroforez Yorum Destek Sistemi")
st.caption("ROI tabanlı Peak Table okuma + HbA hesaplama + varyant kararı")


# -----------------------------
# Tesseract setup
# -----------------------------
def configure_tesseract() -> str:
    env_path = os.getenv("TESSERACT_CMD", "").strip()
    candidates = [
        env_path,
        shutil.which("tesseract"),
        "/usr/bin/tesseract",          # Debian / Streamlit Community Cloud
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract", # Apple Silicon macOS
        "/opt/local/bin/tesseract",
    ]
    for c in candidates:
        if c and Path(c).exists():
            pytesseract.pytesseract.tesseract_cmd = c
            return c
    return ""


TESSERACT_PATH = configure_tesseract()
OCR_AVAILABLE = bool(TESSERACT_PATH)


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


# -----------------------------
# Utilities
# -----------------------------
def pdf_first_page_to_image(file_bytes: bytes) -> Image.Image | None:
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
    ac = ImageOps.autocontrast(gray)
    sharp = ac.filter(ImageFilter.SHARPEN)

    arr = np.array(sharp)
    bin180 = Image.fromarray(np.where(arr > 180, 255, 0).astype(np.uint8))
    bin200 = Image.fromarray(np.where(arr > 200, 255, 0).astype(np.uint8))

    enlarged = sharp.resize((sharp.width * 2, sharp.height * 2))
    return {
        "gray": gray,
        "autocontrast": ac,
        "sharp": sharp,
        "binary180": bin180,
        "binary200": bin200,
        "enlarged": enlarged,
    }


def ocr_text(img: Image.Image, psm=6) -> str:
    if not OCR_AVAILABLE:
        return ""
    cfg = f'--oem 3 --psm {psm}'
    try:
        return pytesseract.image_to_string(img, lang="eng", config=cfg)
    except Exception:
        return ""


def normalize_peak_name(text: str) -> str:
    t = text.strip()
    t = re.sub(r'[|]', ' ', t)
    t = re.sub(r'[^A-Za-z0-9\-/\. ]', '', t)
    t = t.replace("AO", "A0").replace("A O", "A0").replace("A@", "A0")
    t = t.replace("AZ", "A2").replace("A Z", "A2")
    t = t.replace("S Window", "S-Window").replace("SWindow", "S-Window")
    t = t.replace("Unknownn", "Unknown")
    t = re.sub(r'\bAla\b', 'A1a', t)
    t = re.sub(r'\bAlb\b', 'A1b', t)
    return t.strip()


def parse_rows_from_text(text: str) -> pd.DataFrame:
    rows = []
    for raw in text.splitlines():
        line = raw.strip().replace(",", ".")
        if not line:
            continue
        if "peak table" in line.lower() or "total area" in line.lower() or "concentration" in line.lower():
            continue

        # Look for a line ending with area% and starting with peak name
        m = re.search(
            r'(?P<peak>[A-Za-z0-9\-/\. ]+?)\s+'
            r'(?P<rtime>\d+(?:\.\d+)?)\s+'
            r'(?P<height>\d+)\s+'
            r'(?P<area>\d+)\s+'
            r'(?P<area_pct>\d+(?:\.\d+)?)$',
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
    peak_img = roi_img.crop((0, 0, int(w * 0.45), h))
    area_pct_img = roi_img.crop((int(w * 0.76), 0, w, h))

    best_peak_text = ""
    best_pct_text = ""
    best_peak_score = -1
    best_pct_score = -1
    best_peak_img = peak_img
    best_pct_img = area_pct_img

    for name, variant in preprocess_variants(peak_img).items():
        txt = ocr_text(variant, psm=6)
        candidates = [normalize_peak_name(x) for x in txt.splitlines() if x.strip()]
        score = sum(1 for x in candidates if re.search(r'[A-Za-z]', x))
        if score > best_peak_score:
            best_peak_score = score
            best_peak_text = txt
            best_peak_img = variant

    for name, variant in preprocess_variants(area_pct_img).items():
        txt = ocr_text(variant, psm=6)
        cands = [x.strip().replace(",", ".") for x in txt.splitlines() if x.strip()]
        score = sum(1 for x in cands if re.search(r'^\d+(?:\.\d+)?$', x))
        if score > best_pct_score:
            best_pct_score = score
            best_pct_text = txt
            best_pct_img = variant

    peaks = []
    for line in best_peak_text.splitlines():
        n = normalize_peak_name(line)
        if not n:
            continue
        if n.lower() in {"peak", "r.time", "height", "area", "area %", "area%"}:
            continue
        if re.search(r'^\d', n):
            continue
        peaks.append(n)

    pcts = []
    for line in best_pct_text.splitlines():
        line = line.strip().replace(",", ".")
        if re.search(r'^\d+(?:\.\d+)?$', line):
            pcts.append(line)

    max_len = max(len(peaks), len(pcts), 1)
    peaks += [""] * (max_len - len(peaks))
    pcts += [""] * (max_len - len(pcts))

    df = pd.DataFrame({
        "Peak": peaks,
        "R.time": [""] * max_len,
        "Height": [""] * max_len,
        "Area": [""] * max_len,
        "Area %": pcts,
    })
    debug = {
        "peak_img": best_peak_img,
        "peak_text": best_peak_text,
        "pct_img": best_pct_img,
        "pct_text": best_pct_text,
    }
    return df, debug


def merge_candidate_tables(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df1 is not None and not df1.empty:
        return df1
    if df2 is not None and not df2.empty:
        return df2
    return pd.DataFrame(columns=["Peak", "R.time", "Height", "Area", "Area %"])


def standardize_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Peak", "R.time", "Height", "Area", "Area %"])
    out = df.copy()
    for col in ["Peak", "R.time", "Height", "Area", "Area %"]:
        if col not in out.columns:
            out[col] = ""
    out = out[["Peak", "R.time", "Height", "Area", "Area %"]]
    out["Peak"] = out["Peak"].astype(str).apply(normalize_peak_name)
    out["Area %"] = out["Area %"].astype(str).str.replace(",", ".", regex=False)
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
        val = str(row.get("Area %", "")).strip().replace(",", ".")
        try:
            pct = float(val)
        except Exception:
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
    total = 0.0
    included = []
    excluded = []

    for _, row in df.iterrows():
        peak = str(row.get("Peak", "")).strip()
        val = str(row.get("Area %", "")).strip().replace(",", ".")
        try:
            pct = float(val)
        except Exception:
            continue

        peak_low = peak.lower()

        if peak_low == "a2":
            excluded.append((peak, pct, "HbA2 hariç"))
            continue
        if peak_low in {"f", "hbf", "hb f"} or "f-window" in peak_low:
            excluded.append((peak, pct, "HbF hariç"))
            continue
        if "s-window" in peak_low or peak_low == "s":
            if pct > s_threshold:
                excluded.append((peak, pct, f"S-Window > %{s_threshold}"))
                continue
        if "unknown" in peak_low:
            if pct > unknown_threshold:
                excluded.append((peak, pct, f"Unknown > %{unknown_threshold}"))
                continue

        total += pct
        included.append((peak, pct))

    return round(total, 1), included, excluded


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


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Sistem durumu")
    st.write(f"Tesseract: {'Bulundu' if TESSERACT_PATH else 'Bulunamadı'}")
    if TESSERACT_PATH:
        st.code(TESSERACT_PATH)
    st.write(f"pdf2image: {'Hazır' if PDF2IMAGE_OK else 'Yok'}")
    st.write(f"PyMuPDF: {'Hazır' if PYMUPDF_OK else 'Yok'}")
    st.write(f"streamlit-cropper: {'Hazır' if CROPPER_OK else 'Yok'}")
    if not OCR_AVAILABLE:
        st.warning("Tesseract bulunamadı. OCR pasif; tabloyu manuel yapıştırma veya elle düzeltme ile devam edebilirsiniz.")

    st.subheader("Kurallar")
    threshold = st.number_input("S-Window / Unknown varyant eşiği (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    repeat_confirmed = st.checkbox("Tekrar çalışmada benzer bulundu / analitik hata dışlandı", value=True)
    show_debug = st.checkbox("OCR debug göster", value=True)


# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("PDF veya görüntü yükleyin", type=["pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"])

base_img = None
if uploaded is not None:
    base_img, raw_bytes = load_image(uploaded)

if base_img is None:
    st.info("Başlamak için PDF veya görüntü yükleyin.")
    st.stop()

st.subheader("Yüklenen sayfa / görüntü")
st.image(base_img, caption="Kaynak görsel", use_container_width=True)

# -----------------------------
# ROI selection
# -----------------------------
st.subheader("Peak Table ROI seçimi")

roi_img = None

if CROPPER_OK:
    st.info("Fare ile Peak Table alanını seçin.")
    roi_img = st_cropper(
        base_img,
        realtime_update=True,
        box_color="#4F8BF9",
        aspect_ratio=None,
        return_type="image"
    )
else:
    st.warning("İnteraktif cropper bulunamadı. Slaytlarla ROI seçimi kullanılıyor.")
    w, h = base_img.size
    default_left = int(w * 0.08)
    default_top = int(h * 0.55)
    default_right = int(w * 0.82)
    default_bottom = int(h * 0.95)

    c1, c2 = st.columns(2)
    with c1:
        left = st.slider("Sol", 0, w - 1, default_left)
        top = st.slider("Üst", 0, h - 1, default_top)
    with c2:
        right = st.slider("Sağ", 1, w, default_right)
        bottom = st.slider("Alt", 1, h, default_bottom)

    if right <= left:
        right = min(w, left + 10)
    if bottom <= top:
        bottom = min(h, top + 10)

    roi_img = base_img.crop((left, top, right, bottom))

st.image(roi_img, caption="Seçilen Peak Table ROI", use_container_width=True)

# -----------------------------
# OCR and parse
# -----------------------------
st.subheader("ROI OCR ve Peak Table ayrıştırma")

best_text = ""
best_score = -1
best_variant_name = ""
best_variant_img = None

variant_images = preprocess_variants(roi_img)
if OCR_AVAILABLE:
    for name, img_variant in variant_images.items():
        txt = ocr_text(img_variant, psm=6)
        df_candidate = parse_rows_from_text(txt)
        score = len(df_candidate)
        if score > best_score:
            best_score = score
            best_text = txt
            best_variant_name = name
            best_variant_img = img_variant
    parsed_direct = standardize_table(parse_rows_from_text(best_text))
    parsed_roi_columns, roi_debug = parse_rows_by_roi_columns(roi_img)
    parsed_roi_columns = standardize_table(parsed_roi_columns)
else:
    parsed_direct = standardize_table(pd.DataFrame())
    parsed_roi_columns = standardize_table(pd.DataFrame())
    roi_debug = {"peak_img": roi_img, "peak_text": "", "pct_img": roi_img, "pct_text": ""}

parsed_df = merge_candidate_tables(parsed_direct, parsed_roi_columns)
parsed_df = standardize_table(parsed_df)

if show_debug:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**Seçilen tam ROI OCR ön işlemi:** `{best_variant_name}`")
        if best_variant_img is not None:
            st.image(best_variant_img, caption="Tam ROI OCR görseli", use_container_width=True)
        st.text_area("Tam ROI OCR metni", value=best_text, height=200)
    with d2:
        st.markdown("**ROI sütun ayrımı debug**")
        st.image(roi_debug["peak_img"], caption="Peak kolonu", use_container_width=True)
        st.text_area("Peak OCR", value=roi_debug["peak_text"], height=120)
        st.image(roi_debug["pct_img"], caption="Area % kolonu", use_container_width=True)
        st.text_area("Area % OCR", value=roi_debug["pct_text"], height=120)

st.dataframe(parsed_df, use_container_width=True)

# -----------------------------
# Manual correction
# -----------------------------
st.subheader("Peak Table doğrulama / manuel düzeltme")
edited_df = st.data_editor(
    parsed_df if not parsed_df.empty else pd.DataFrame(
        [{"Peak": "", "R.time": "", "Height": "", "Area": "", "Area %": ""}]
    ),
    num_rows="dynamic",
    use_container_width=True,
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
        parts = re.split(r'[\t;]+', line.strip())
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
        edited_df = pd.DataFrame(rows)
        st.info("Manuel yapıştırılan metin uygulandı.")
        st.dataframe(edited_df, use_container_width=True)

edited_df = standardize_table(pd.DataFrame(edited_df))

# -----------------------------
# Calculation
# -----------------------------
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
    st.metric("HbA2", "-" if named["HbA2"] is None else f"{named['HbA2']:.1f}%")
with col3:
    st.metric("HbF", "-" if named["HbF"] is None else f"{named['HbF']:.1f}%")
with col4:
    st.metric("Varyant lehine bulgu", "Evet" if is_variant else "Hayır")

st.write("**Varyant gerekçesi:** " + (", ".join(variant_reasons) if variant_reasons else "Yok"))

with st.expander("HbA hesaplama detayları"):
    st.write("**HbA'ya dahil edilen pikler**")
    st.dataframe(pd.DataFrame(included_peaks, columns=["Peak", "Area %"]), use_container_width=True)
    st.write("**HbA dışında bırakılan pikler**")
    if excluded_peaks:
        st.dataframe(pd.DataFrame(excluded_peaks, columns=["Peak", "Area %", "Neden"]), use_container_width=True)
    else:
        st.write("Hariç tutulan pik yok.")

# -----------------------------
# Final interpretation inputs
# -----------------------------
st.subheader("Klinik girişler")
c1, c2 = st.columns(2)
with c1:
    age_years = st.number_input("Yaş (yıl)", min_value=0.0, max_value=120.0, value=18.0, step=0.1)
with c2:
    sex = st.selectbox("Cinsiyet", ["Kadın", "Erkek", "Belirtilmedi"])

manual_override_variant = st.checkbox("Varyant lehine bulgu olarak manuel işaretle", value=False)

if st.button("Klinik yorumu üret", type="primary"):
    final_variant = is_variant or manual_override_variant
    hba2 = named["HbA2"] if named["HbA2"] is not None else 0.0
    hbf = named["HbF"] if named["HbF"] is not None else 0.0

    if final_variant:
        result = {
            "block": None,
            "reason": "S-Window ve/veya Unknown pik > %5 olduğu için ek pik / yapısal varyant lehine bulgu",
            "comment": build_variant_comment(named, repeat_confirmed=repeat_confirmed),
        }
    else:
        result = generate_comment(
            age_years=age_years,
            sex=sex,
            hba2=float(hba2),
            hbf=float(hbf),
            hba=float(calc_hba),
            extra_variant_peak=False
        )

    st.subheader("Sonuç")
    if result["block"]:
        st.success(f"Seçilen blok: {result['block']}")
    else:
        st.warning("Varyant yolu kullanıldı veya standart blok seçilemedi")

    st.write(f"**Gerekçe:** {result['reason']}")
    st.text_area("Oluşan yorum", value=result["comment"], height=320)

    export_text = (
        "Hb ELEKTROFOREZ YORUM RAPORU\n"
        "===========================\n\n"
        f"Yaş: {age_years}\n"
        f"Cinsiyet: {sex}\n"
        f"HbA (hesaplanan): {calc_hba:.1f}\n"
        f"HbA2: {('-' if named['HbA2'] is None else f'{named['HbA2']:.1f}')}\n"
        f"HbF: {('-' if named['HbF'] is None else f'{named['HbF']:.1f}')}\n"
        f"S-Window: {('-' if named['S-Window'] is None else f'{named['S-Window']:.1f}')}\n"
        f"Unknown: {('-' if named['Unknown'] is None else f'{named['Unknown']:.1f}')}\n"
        f"Varyant lehine bulgu: {'Evet' if final_variant else 'Hayır'}\n\n"
        "Peak Table\n"
        "----------\n"
        f"{df_to_text(edited_df)}\n\n"
        "HbA hesaplama kuralı\n"
        "--------------------\n"
        "HbA = HbA2 ve HbF hariç tüm piklerin toplamı; S-Window ve Unknown yalnızca %5'in altındaysa HbA'ya dahil edilir.\n\n"
        f"Gerekçe: {result['reason']}\n\n"
        "Yorum\n"
        "-----\n"
        f"{result['comment']}\n"
    )

    st.download_button(
        "Raporu TXT olarak indir",
        data=export_text,
        file_name="hb_elektroforez_roi_yorum_raporu.txt",
        mime="text/plain"
    )

with st.expander("Notlar"):
    st.markdown("""
- Bu sürüm PDF zorunlu değildir; doğrudan taranmış görsel de yüklenebilir.
- Peak Table alanını ROI ile seçmeniz beklenir.
- Varyant lehine bulgu kuralı:
  - S-Window > %5
  - Unknown > %5
- HbA hesaplama kuralı:
  - HbA2 ve HbF hariç diğer tüm piklerin toplamı
  - S-Window ve Unknown yalnızca %5'in altındaysa HbA'ya dahil edilir
- OCR hatası olasılığı nedeniyle tabloyu gözle doğrulayıp düzeltmeniz önerilir.
- Tesseract kurulmamış ortamlarda OCR yerine manuel tablo girişiyle uygulama kullanılabilir.
""")
