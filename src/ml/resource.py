from __future__ import annotations
import json
from pathlib import Path

def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file())

def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def fmt(x: float, nd=4) -> str:
    return f"{x:.{nd}f}"

def main():
    data_root = Path("data/fall_dataset")
    results_dir = data_root / "results"
    metrics_path = results_dir / "metrics.json"
    images_dir = data_root / "images"

    if not metrics_path.exists():
        raise SystemExit(f"Tidak menemukan {metrics_path}")

    m = json.loads(metrics_path.read_text(encoding="utf-8"))

    # 1) Dataset ringkas
    n_fall = count_images(images_dir / "fall")
    n_not = count_images(images_dir / "not-fall")
    total = n_fall + n_not
    p_fall = (n_fall / total * 100) if total else 0.0
    p_not  = (n_not  / total * 100) if total else 0.0

    device = m.get("device", "N/A")

    # 2) Metrik global
    acc    = m.get("accuracy", 0.0)
    bal    = m.get("balanced_accuracy", 0.0)
    mcc    = m.get("mcc", 0.0)
    kappa  = m.get("kappa", 0.0)

    # 2b) Metrik per kelas
    per_class = m.get("per_class", [])
    # Buat dict: class_name -> metrics
    per_cls = {c["class"]: c for c in per_class if "class" in c}

    # 3) Confusion details
    conf_details = m.get("confusion_details", [])
    conf_map = {c["class"]: c for c in conf_details if "class" in c}

    # 4) ROC/PR
    is_binary = "roc_auc" in m or "avg_precision" in m or "best_threshold" in m
    roc_auc = m.get("roc_auc", None)
    ap = m.get("avg_precision", None)
    best_thr = m.get("best_threshold", None)
    roc_ovr = m.get("roc_auc_ovr", {})
    ap_ovr  = m.get("avg_precision_ovr", {})

    # --------- Susun paragraf ----------
    # 1. Deskripsi Dataset dan Protokol
    split_info = (
        "Berdasarkan struktur data saat ini, tidak terdapat pemisahan eksplisit train/val/test; "
        "evaluasi dilakukan pada keseluruhan folder images."
    )

    sec1 = []
    sec1.append("1. Deskripsi Dataset dan Protokol Eksperimen")
    sec1.append(
        f"Dataset terdiri dari dua kelas, yakni fall dan not-fall, dengan total {total} citra "
        f"(fall: {n_fall}; not-fall: {n_not}; proporsi: fall {fmt_pct(p_fall)}, not-fall {fmt_pct(p_not)}). "
        "Citra berukuran 224×224 piksel sesuai konfigurasi evaluasi. "
        f"Model dievaluasi pada perangkat {device}. {split_info}"
    )
    sec1.append(
        "Metrik utama yang dilaporkan meliputi Accuracy, Balanced Accuracy, Matthews Correlation Coefficient (MCC), "
        "dan Cohen’s Kappa untuk mengakomodasi potensi ketidakseimbangan kelas."
    )

    # 2. Hasil Kuantitatif Utama
    sec2 = []
    sec2.append("2. Hasil Kuantitatif Utama")
    sec2.append(
        f"Secara keseluruhan, model mencapai Accuracy {fmt(acc)}, Balanced Accuracy {fmt(bal)}, "
        f"MCC {fmt(mcc)}, dan Cohen’s Kappa {fmt(kappa)}."
    )
    if per_cls:
        # Tampilkan semua kelas yang tersedia
        per_cls_lines = []
        for cname, cmet in per_cls.items():
            per_cls_lines.append(
                f"- Kelas {cname}: precision {fmt(cmet.get('precision', 0.0), 4)}, "
                f"recall/sensitivity {fmt(cmet.get('recall', 0.0), 4)}, "
                f"F1 {fmt(cmet.get('f1', 0.0), 4)}, support {cmet.get('support', 0)}"
            )
        sec2.append("Metrik per kelas:")
        sec2.extend(per_cls_lines)

    # 3. Analisis Confusion Matrix dan TP/TN/FP/FN
    sec3 = []
    sec3.append("3. Analisis Confusion Matrix dan TP/TN/FP/FN")
    if conf_map:
        for cname, c in conf_map.items():
            sec3.append(
                f"Untuk kelas {cname}: TP {c.get('tp',0)}, FP {c.get('fp',0)}, "
                f"FN {c.get('fn',0)}, TN {c.get('tn',0)}; "
                f"sensitivity {fmt(c.get('sensitivity',0.0),4)}, "
                f"specificity {fmt(c.get('specificity',0.0),4)}."
            )
        sec3.append(
            "Kesalahan FN pada kelas fall lebih kritis karena berpotensi melewatkan kejadian jatuh, "
            "sementara FP meningkatkan frekuensi alarm palsu."
        )
    else:
        sec3.append("Rincian TP/TN/FP/FN per kelas tidak tersedia pada evaluasi ini.")

    # 4. Kurva ROC, Precision–Recall, dan Threshold
    sec4 = []
    sec4.append("4. Kurva ROC, Precision–Recall, dan Threshold")
    if is_binary:
        sec4.append(
            f"Pada skenario biner, Area Under ROC (AUC) {fmt(roc_auc or 0.0)}, "
            f"Average Precision (AP) {fmt(ap or 0.0)}, dan ambang optimal (Youden J) {fmt(best_thr or 0.5, 3)}."
        )
        sec4.append(
            "Pemilihan ambang di bawah nilai optimal dapat meningkatkan recall (menurunkan FN) "
            "dengan konsekuensi kenaikan FP; sebaliknya, menaikkan ambang dapat menurunkan FP "
            "namun berisiko meningkatkan FN."
        )
    elif roc_ovr or ap_ovr:
        if roc_ovr:
            sec4.append("AUC ROC one-vs-rest per kelas:")
            for cname, v in roc_ovr.items():
                sec4.append(f"- {cname}: AUC {fmt(v)}")
        if ap_ovr:
            sec4.append("Average Precision (PR) one-vs-rest per kelas:")
            for cname, v in ap_ovr.items():
                sec4.append(f"- {cname}: AP {fmt(v)}")
    else:
        sec4.append("Artefak ROC/PR tidak tersedia pada evaluasi ini.")

    # 5. Kalibrasi dan Kepercayaan Prediksi
    sec5 = []
    sec5.append("5. Kalibrasi dan Kepercayaan Prediksi")
    sec5.append(
        "Kalibrasi probabilitas belum dievaluasi pada tahap ini. "
        "Disarankan menambahkan penilaian Expected Calibration Error (ECE) atau Brier Score "
        "serta diagram reliabilitas untuk menilai kesesuaian skor probabilitas dengan frekuensi kejadian. "
        "Kalibrasi yang baik mempermudah penetapan ambang dan kebijakan eskalasi."
    )

    # Gabungkan
    lines = []
    for sec in (sec1, sec2, sec3, sec4, sec5):
        lines.append("## " + sec[0])
        lines.extend(sec[1:])
        lines.append("")  # blank line

    out_path = results_dir / "journal_hasil_pembahasan.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Selesai. Tersimpan: {out_path}")

if __name__ == "__main__":
    main()