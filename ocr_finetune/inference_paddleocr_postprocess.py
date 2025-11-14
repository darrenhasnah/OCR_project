from pathlib import Path
import argparse, re
import numpy as np
import cv2
import os

from paddleocr import PaddleOCR
import logging

ROOT = Path(r"d:/uph/Semester 7/Grafika Komputer/UAS/UAS_Grafika")
DATA_DIR = ROOT / "testing"  # default images folder
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Normalisasi dulu
COMMON_CORRECTIONS = {
    "\\|": "1", "I": "1", "l": "1", 
    "O": "0", "o": "0",          
    "S": "5", "s": "5",              
    "B": "8", "—": "-", "¢": "c",
}
ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.,:/()%$+ RpIDR")

def strip_disallowed(s: str) -> str:
    """Remove characters outside the allowed set."""
    return "".join(ch for ch in s if ch in ALLOWED_CHARS or ch.isspace())

def correct_text_token(tok: str) -> str:
    """Conservative token-level corrections (digit/char confusions)."""
    if re.search(r"\d", tok):
        for k, v in COMMON_CORRECTIONS.items():
            tok = tok.replace(k, v)
        tok = re.sub(r"[^0-9A-Za-z\.,:;\$%\-\/\(\) ]+", "", tok)
    return tok

def normalize_line(line: str) -> str:
    """Clean and normalize a single OCR line."""
    line = line.strip()
    line = strip_disallowed(line)
    line = re.sub(r"[|]{2,}", "|", line)
    line = re.sub(r"\s{2,}", " ", line)
    parts = [correct_text_token(p) for p in line.split()]
    line = " ".join(parts)
    # Normalisasi lanjutan
    repl = [
        (r"\bsub\s*total\b", "SUB TOTAL"),
        (r"\bsubtotal\b", "SUB TOTAL"),
        (r"\bgrand\s*total\b", "GRAND TOTAL"),
        (r"\btotal\b", "TOTAL"),
        (r"\bppn\b", "PPN"),
        (r"\btax\b", "TAX"),
        (r"\bpajak\b", "PAJAK"),
    ]
    for pat, rep in repl:
        line = re.sub(pat, rep, line, flags=re.I)
    return line

# Image pre-processing
def preprocess_for_ocr(img_path_or_array):
    """Load image, convert to gray, upscale and apply CLAHE."""
    try:
        if isinstance(img_path_or_array, (str, Path)):
            img = cv2.imread(str(img_path_or_array))
        else:
            img = img_path_or_array
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if max(h, w) < 1100:
            gray = cv2.resize(gray, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray
    except Exception:
        return None

# Normalisasi angka harga
AMOUNT_RE = r"(?<!\w)(?:Rp|IDR|\$)?\s?(\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|\d+[.,]\d{2})(?!\w)"
NPWP_RE = re.compile(r"\bNPWP\b", re.I)
NOISE_LINE_RE = re.compile(r"\b(P81|P8I|P8l|npwp)\b", re.I)

def norm_money_str(s: str) -> str:
    """Normalize money-like strings into a dot-decimal float string."""
    if not s:
        return ""
    s = s.strip().replace(" ", "")
    s = re.sub(r"^(Rp|IDR|\$)", "", s, flags=re.I)
    if re.search(r"\d+\.\d{3},\d{2}$", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.search(r"^\d{1,3}\.\d{3}$", s) and s.count(",") == 0:
        s = s.replace(".", "")
    elif re.search(r"\d+,\d{2}$", s) and s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        if s.count(".") >= 2 and s.count(",") == 0:
            s = s.replace(".", "")
        s = s.replace(",", "")
    return s

def to_float(s: str):
    try:
        return float(norm_money_str(s))
    except Exception:
        return None

def find_amounts(line: str):
    res = [m.group(1) for m in re.finditer(AMOUNT_RE, line)]
    if res:
        return res
    # cek Rp berapa
    simple = []
    for m in re.finditer(r"(?<!\w)(?:Rp|IDR)\s?(\d+)(?!\w)", line, flags=re.I):
        simple.append(m.group(1))
    return simple
def find_amount_near(lines, idx, window=2, prefer_prev=False):
    """Cari teks jumlah uang di sekitar baris idx.
Balikin kandidat yang paling cocok (amount_str, float_value) atau (None, None).
Caranya: kumpulin semua kandidat di area window (skip yang kayak noise). Kalau ada di baris yang sama, pilih yang paling kanan di baris itu.
Kalau nggak ada, pilih kandidat dengan nilai angka paling gede (biasanya total itu yang angkanya paling besar di sekitar situ)
    """
    candidates = []
    for j in range(max(0, idx - window), min(len(lines), idx + window + 1)):
        ln = lines[j]
        if NPWP_RE.search(ln) or NOISE_LINE_RE.search(ln):
            continue
        amts = find_amounts(ln)
        if not amts:
            continue
        for a in amts:
            v = to_float(a)
            if v is not None:
                candidates.append((a, v, j))
    if not candidates:
        return None, None
    same_line = [c for c in candidates if c[2] == idx]
    if same_line:
        best = max(same_line, key=lambda t: t[1])
        return best[0], best[1]
    # angka total lebih sering di line "previous" jadi kita ambil itu
    if prefer_prev:
        prev_line = [c for c in candidates if c[2] == idx - 1]
        if prev_line:
            best = max(prev_line, key=lambda t: t[1])
            return best[0], best[1]
        next_line = [c for c in candidates if c[2] == idx + 1]
        if next_line:
            best = max(next_line, key=lambda t: t[1])
            return best[0], best[1]
    else:
        next_line = [c for c in candidates if c[2] == idx + 1]
        if next_line:
            best = max(next_line, key=lambda t: t[1])
            return best[0], best[1]
        prev_line = [c for c in candidates if c[2] == idx - 1]
        if prev_line:
            best = max(prev_line, key=lambda t: t[1])
            return best[0], best[1]
   # sebalikny ambil yg terbesar
    best = max(candidates, key=lambda t: t[1])
    return best[0], best[1]


def max_amount(lines):
    m_s, m_v = None, None
    for ln in lines:
        # skip noisy lines
        if NPWP_RE.search(ln) or NOISE_LINE_RE.search(ln):
            continue
        for a in find_amounts(ln):
            v = to_float(a)
            if v is not None and (m_v is None or v > m_v):
                m_s, m_v = a, v
    return m_s, m_v

KEY_TOTAL = re.compile(r"\b(GRAND\s+TOTAL|TOTAL\s*RP|TOTAL)\b", re.I)
KEY_SUB   = re.compile(r"\b(SUB\s*TOTAL|JUMLAH)\b", re.I)
KEY_TAX   = re.compile(r"\b(TAX|PPN|PAJAK)\b", re.I)
KEY_SVC   = re.compile(r"\b(SERVICE|SERVIS)\b", re.I)
KEY_CASH  = re.compile(r"\b(CASH|TUNAI|BAYAR|PAID)\b", re.I)
KEY_CHG   = re.compile(r"\b(CHANGE|KEMBALIAN|KEMBALI)\b", re.I)
KEY_GIVEN = re.compile(r"\b(GIVEN|GIVEN:|PAID|BAYAR)\b", re.I)
KEY_KEMBALI = re.compile(r"\b(KEMBALI|KEMBALIAN|KEMBALI\b|KEMBALI:)\b", re.I)


def extract_items(lines):
    """Ngambil item dengan ngelompokin nama/jumlah/harga dari baris-baris yang saling deket.

Fungsi ini nge scan tiap baris dan nyari baris nama (yang nggak ada angka uangnya), terus cocokin sama baris jumlah atau harga yang muncul dalam 2 baris setelahnya.

Juga nge-handle kasus kalau harga muncul duluan, terus nama item-nya ada di baris atasnya.
    """
    items = []
    n = len(lines)
    i = 0
    skip_prefix = re.compile(r"^(Jl\.|Jalan|http:|https:|www\.|\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2})", re.I)
    qty_pattern = re.compile(r"(\d+)\s*[xX]\s*(?:Rp\s*)?([\d\.,]+)")
    qty_pattern_alt = re.compile(r"(\d+)\s+Rp\s*([\d\.,]+)")
    while i < n:
        ln = lines[i].strip()
        if not ln or KEY_TOTAL.search(ln) or KEY_SUB.search(ln) or KEY_TAX.search(ln) or "NPWP" in ln.upper() or skip_prefix.search(ln):
            i += 1
            continue

        amts_here = find_amounts(ln)
        if amts_here:

            name = None
            j = i - 1
            while j >= 0:
                cand = lines[j].strip()
                if cand and not find_amounts(cand) and not KEY_TOTAL.search(cand) and not KEY_SUB.search(cand) and "NPWP" not in cand.upper():
                    name = cand
                    break
                j -= 1

            # cek qty lines (e.g., '2 x Rp25.000')
            qty = 1
            unit_price = None
            # ambil next line buat qty
            if i + 1 < n:
                m = qty_pattern.search(lines[i+1]) or qty_pattern_alt.search(lines[i+1])
                if m:
                    qty = int(m.group(1))
                    unit_price = to_float(m.group(2))
                    i += 1

            total_price = to_float(amts_here[-1])
            if unit_price and total_price and abs(unit_price * qty - total_price) < 1.0:
                price_val = total_price
            elif unit_price and not total_price:
                price_val = unit_price * qty
            else:
                price_val = total_price

            if name and price_val and price_val > 0:
                items.append({"item": name, "qty": qty, "price": f"{price_val:,.2f}"})
                i += 1
                continue

        # if current line looks like a name, check next two lines for amount/qty
        amts_next = []
        qty = 1
        unit_price = None
        found_price = None
        for k in (i+1, i+2):
            if k >= n: break
            s = lines[k]
            # qty pattern
            m = qty_pattern.search(s) or qty_pattern_alt.search(s)
            if m and not unit_price:
                qty = int(m.group(1))
                unit_price = to_float(m.group(2))
                continue
            a = find_amounts(s)
            if a:
                found_price = to_float(a[-1])
                # if we already have unit_price, prefer total that matches
                break

        if found_price and (not KEY_TOTAL.search(ln)):
        
            name = ln
            if unit_price and found_price and abs(unit_price * qty - found_price) < 1.0:
                price_val = found_price
            elif unit_price and not found_price:
                price_val = unit_price * qty
            else:
                price_val = found_price
            if price_val and price_val > 0:
                items.append({"item": name, "qty": qty, "price": f"{price_val:,.2f}"})
                i = k + 1
                continue

        i += 1

    return items

# Function to extract subtotal, service, tax, total, given, and change
def extract_summary(ocr_text):
    summary = {}
    # Work line-by-line and search nearby amounts for labels
    lines = [ln for ln in ocr_text.splitlines() if ln.strip()]

    sub_s, sub_v = None, None
    svc_s, svc_v = None, None
    tax_s, tax_v = None, None
    total_s, total_v = None, None
    cash_s, cash_v = None, None
    chg_s, chg_v = None, None

    for i, ln in enumerate(lines):
        if KEY_SUB.search(ln) and sub_s is None:
            a, v = find_amount_near(lines, i, window=2)
            if a:
                sub_s, sub_v = a, v
        if KEY_SVC.search(ln) and svc_s is None:
            a, v = find_amount_near(lines, i, window=2)
            if a:
                svc_s, svc_v = a, v
        if KEY_TAX.search(ln) and tax_s is None:
            a, v = find_amount_near(lines, i, window=2)
            if a:
                tax_s, tax_v = a, v
        if KEY_TOTAL.search(ln) and total_s is None:
            a, v = find_amount_near(lines, i, window=2, prefer_prev=True)
            if a:
                total_s, total_v = a, v
        if KEY_CASH.search(ln) and cash_s is None:
            a, v = find_amount_near(lines, i, window=2)
            if a:
                cash_s, cash_v = a, v
        if KEY_CHG.search(ln) and chg_s is None:
            a, v = find_amount_near(lines, i, window=2)
            if a:
                chg_s, chg_v = a, v
        if KEY_GIVEN.search(ln) and cash_s is None:
            a, v = find_amount_near(lines, i, window=2)
            if a:
                cash_s, cash_v = a, v
        if KEY_KEMBALI.search(ln) and chg_s is None:
            a, v = find_amount_near(lines, i, window=2)
            if a:
                chg_s, chg_v = a, v

    # Fallback logic: use document amounts to reason about total vs subtotal/given
    items = extract_items(ocr_text)
    max_item = max([to_float(it['price']) for it in items], default=0.0)
    # choose largest amount in document (excluding NPWP/noise)
    cand_s, cand_v = max_amount(lines)
    # If total missing or clearly too small (smaller than subtotal or max item), prefer largest amount
    if cand_v:
        if total_v is None or total_v < max_item or (sub_v is not None and total_v < sub_v):
            # prefer an amount larger than subtotal/item
            if cand_v and (sub_v is None or cand_v >= sub_v):
                total_s, total_v = cand_s, cand_v
    # If subtotal looks wrong (e.g., equals a very large number from NPWP), try to pick a reasonable subtotal
    if sub_v is None or (max_item and sub_v < max_item):
        # attempt to pick an amount smaller than total but larger than items sum
        if cand_v and (total_v is not None and cand_v < total_v and cand_v >= max_item):
            sub_s, sub_v = cand_s, cand_v

    # assemble summary strings (preserve original formatting where possible)
    def fmt(a):
        try:
            return f"{to_float(a):,.2f}" if a else ""
        except Exception:
            return a or ""

    summary = {
        "subtotal": fmt(sub_s),
        "service": fmt(svc_s),
        "tax": fmt(tax_s),
        "total": fmt(total_s),
        "given": fmt(cash_s),
        "change": fmt(chg_s),
        "items": items,
    }

    return summary

# ===================== MAIN =====================
def run(limit=5, lang="en", min_conf=0.30):
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    imgs = sorted([p for p in DATA_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not imgs:
        print("No images found in testing folder")
        return

    for p in imgs[:limit]:
        print(f"Processing {p.name}...")

        img_proc = preprocess_for_ocr(str(p))
        img_input = img_proc
        if img_input is not None and img_input.ndim == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
        if img_input is None:
            img_input = str(p)

        try:
            res = ocr.predict(img_input)
        except Exception as e:
            print(f"PaddleOCR error for {p.name}: {e}")
            continue

        paired = []
        try:
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], list) and isinstance(res[1], list):
                dets, recs = res
                for b, r in zip(dets, recs):
                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                        text = r[0]; conf = float(r[1])
                    else:
                        text = r; conf = 1.0
                    paired.append((b, (text, conf)))
            elif isinstance(res, list) and res and isinstance(res[0], dict):
                for entry in res:
                    rec_texts = entry.get("rec_texts") or entry.get("rec_text") or []
                    rec_scores = entry.get("rec_scores") or entry.get("rec_score") or []
                    rec_polys  = entry.get("rec_polys")  or entry.get("rec_boxes") or entry.get("dt_polys") or []
                    n = min(len(rec_texts), len(rec_scores), len(rec_polys)) if rec_texts and rec_scores and rec_polys else 0
                    for i in range(n):
                        b = rec_polys[i]
                        t = rec_texts[i]
                        s = rec_scores[i]
                        try:
                            box_pts = b.tolist() if hasattr(b, "tolist") else list(b)
                        except Exception:
                            box_pts = b
                        paired.append((box_pts, (t, s)))
        except Exception:
            paired = []

        # Bikin daftar baris mentah dari pasangan (box, text)
        rows = []
        for entry in paired:
            try:
                box = entry[0]; text = entry[1][0]; conf = float(entry[1][1])
            except Exception:
                continue
            if conf < min_conf or not isinstance(text, str):
                continue
            try:
                xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
            except Exception:
                xs = [0]; ys = [0]
            rows.append((min(ys), min(xs), text.strip(), conf))

        rows.sort(key=lambda t: (round(t[0] / 8), t[1]))
        raw_lines = [r[2] for r in rows]

        norm_lines = [normalize_line(ln) for ln in raw_lines if ln.strip()]

        # items butuh list baris yang udah dinormalisasi
        # ambil summary dulu (kadang summary pake info items), lalu parse items
        summary = extract_summary("\n".join(norm_lines))
        items = extract_items(norm_lines)

    # Terapkan override dari pola 'N x RpU' kalau ada
    # Berguna kalo parser salah attach harga ke baris yang salah
        def apply_qty_overrides(lines, items):
            # bikin map: nama item (normal) -> entry item
            idx = {}
            for it in items:
                key = re.sub(r"\s+", " ", it.get("item","").strip().lower())
                idx[key] = it

            n = len(lines)
            qty_re = re.compile(r"^(\d+)\s*[xX]\s*(?:Rp\s*)?([\d\.,]+)")
            for i in range(n-1):
                m = qty_re.search(lines[i+1])
                if not m:
                    continue
                # nama kandidat biasanya baris sekarang kalo nggak ada angka
                name_cand = lines[i].strip()
                if not name_cand or find_amounts(name_cand):
                    # coba baris sebelumnya yang gak kosong
                    j = i-1
                    while j >= 0:
                        if lines[j].strip() and not find_amounts(lines[j]):
                            name_cand = lines[j].strip(); break
                        j -= 1
                if not name_cand:
                    continue
                qty = int(m.group(1)); unit = to_float(m.group(2))
                if unit is None:
                    continue
                key = re.sub(r"\s+", " ", name_cand.strip().lower())
                # kalo ada item yang cocok (atau mirip), override qty & total
                if key in idx:
                    idx[key]["qty"] = qty
                    idx[key]["price"] = f"{unit * qty:,.2f}"
                else:
                    # coba match fuzzy: check prefix / kata pertama
                    for k in list(idx.keys()):
                        if k.startswith(key) or key.startswith(k) or key.split()[0] == k.split()[0]:
                            idx[k]["qty"] = qty
                            idx[k]["price"] = f"{unit * qty:,.2f}"
                            break
            return list(idx.values())

        items = apply_qty_overrides(norm_lines, items)

    # Post-process items:
    # - buang baris yang bukan item (POS/TUNAI/dsb)
    # - buang harga yang kebetulan sama ama total/given/change
    # - gabung duplicate (sum qty + total)
        def group_and_filter_items(items, summary):
            filtered = []
            # kata-kata yang mau kita skip
            excl_kw = ["POS", "TUNAI", "TOTAL", "KEMBALI", "BAYAR", "PESANAN", "KARYAWAN", "LINK", "KRABAT", "THANK"]
            tot = to_float(summary.get("total") or "0")
            given = to_float(summary.get("given") or "0")
            chg = to_float(summary.get("change") or "0")
            for it in items:
                name = it.get("item", "").strip()
                if not name or len(name) < 2:
                    continue
                up = name.upper()
                if any(k in up for k in excl_kw):
                    continue
                price = to_float(it.get("price") or "0")
                if price is None:
                    continue
                # skip kalo harganya kebetulan sama ama total/given/change (biasanya salah)
                if (tot and abs(price - tot) < 1.0) or (given and abs(price - given) < 1.0) or (chg and abs(price - chg) < 1.0):
                    continue
                filtered.append({"item": name, "qty": int(it.get("qty", 1)), "total": price})
            # gabung berdasarkan nama yang dinormalisasi
            aggr = {}
            for it in filtered:
                key = re.sub(r"\s+", " ", it["item"].strip().lower())
                if key in aggr:
                    aggr[key]["qty"] += it["qty"]
                    aggr[key]["total"] += it["total"]
                else:
                    aggr[key] = {"item": it["item"].strip(), "qty": it["qty"], "total": it["total"]}
            # hasil akhir: list item dengan total price yang diformat
            out = []
            for k, v in aggr.items():
                out.append({"item": v["item"], "qty": v["qty"], "price": f"{v['total']:,.2f}"})
            return out

        items = group_and_filter_items(items, summary)

        print(f"--- {p.name} ---")
        print("Cleaned OCR:")
        print("\n".join(norm_lines))
        print("\n" + "=" * 30 + "\n")
        print("Summary:")
        print(f"  Subtotal : {summary.get('subtotal', 'Not found')}")
        print(f"  Service  : {summary.get('service', 'Not found')}")
        print(f"  Total    : {summary.get('total', 'Not found')}")
        print(f"  Given    : {summary.get('given', 'Not found')}")
        print(f"  Change   : {summary.get('change', 'Not found')}")
        print("Items:")
        if items:
            for item in items:
                print(f"  - ({item.get('qty',1)}) {item['item']}  -> {item['price']}")
        else:
            print("  (not detected)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--min-conf", type=float, default=0.30)
    args = ap.parse_args()

    run(limit=args.limit, lang=args.lang, min_conf=args.min_conf)
