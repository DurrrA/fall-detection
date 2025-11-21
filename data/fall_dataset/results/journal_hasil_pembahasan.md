## 1. Deskripsi Dataset dan Protokol Eksperimen
Dataset terdiri dari dua kelas, yakni fall dan not-fall, dengan total 575 citra (fall: 349; not-fall: 226; proporsi: fall 60.70%, not-fall 39.30%). Citra berukuran 224×224 piksel sesuai konfigurasi evaluasi. Model dievaluasi pada perangkat N/A. Berdasarkan struktur data saat ini, tidak terdapat pemisahan eksplisit train/val/test; evaluasi dilakukan pada keseluruhan folder images.
Metrik utama yang dilaporkan meliputi Accuracy, Balanced Accuracy, Matthews Correlation Coefficient (MCC), dan Cohen’s Kappa untuk mengakomodasi potensi ketidakseimbangan kelas.

## 2. Hasil Kuantitatif Utama
Secara keseluruhan, model mencapai Accuracy 0.9426, Balanced Accuracy 0.9449, MCC 0.8819, dan Cohen’s Kappa 0.8809.
Metrik per kelas:
- Kelas fall: precision 0.9702, recall/sensitivity 0.9341, F1 0.9518, support 349
- Kelas not-fall: precision 0.9038, recall/sensitivity 0.9558, F1 0.9290, support 226

## 3. Analisis Confusion Matrix dan TP/TN/FP/FN
Untuk kelas fall: TP 326, FP 10, FN 23, TN 216; sensitivity 0.9341, specificity 0.9558.
Untuk kelas not-fall: TP 216, FP 23, FN 10, TN 326; sensitivity 0.9558, specificity 0.9341.
Kesalahan FN pada kelas fall lebih kritis karena berpotensi melewatkan kejadian jatuh, sementara FP meningkatkan frekuensi alarm palsu.

## 4. Kurva ROC, Precision–Recall, dan Threshold
Pada skenario biner, Area Under ROC (AUC) 0.9919, Average Precision (AP) 0.9947, dan ambang optimal (Youden J) 0.333.
Pemilihan ambang di bawah nilai optimal dapat meningkatkan recall (menurunkan FN) dengan konsekuensi kenaikan FP; sebaliknya, menaikkan ambang dapat menurunkan FP namun berisiko meningkatkan FN.

## 5. Kalibrasi dan Kepercayaan Prediksi
Kalibrasi probabilitas belum dievaluasi pada tahap ini. Disarankan menambahkan penilaian Expected Calibration Error (ECE) atau Brier Score serta diagram reliabilitas untuk menilai kesesuaian skor probabilitas dengan frekuensi kejadian. Kalibrasi yang baik mempermudah penetapan ambang dan kebijakan eskalasi.
