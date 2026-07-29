"""One module per figure kind; scripts under scripts/ only load data and save.

What:
    每個模組負責一種圖，公開一個 `draw_*` 函式，接收已備妥的資料、回傳
    matplotlib Figure。模組不讀檔、不寫檔、不呼叫 plt.show —— 輸入輸出都在
    呼叫端，模組只描述「這種圖長什麼樣」。

Why:
    繪圖邏輯散在 scripts/ 的 30 餘支腳本裡時，同一種圖會被重寫多次，樣式與
    座標軸慣例隨之漂移，且無法在不產生檔案的前提下測試。把圖形本身收成模組
    後，scripts/ 的對應腳本退化成薄 CLI（讀資料 → draw → save_figure），
    重複自然消失，樣式由 pi_con.plot_style 單點控制。

Convention:
    - `draw_*(data...) -> Figure`：純函式，不碰檔案系統
    - 呼叫端負責 `apply_journal_rcparams()` 與 `save_figure()`
    - 顏色、線型、圖寬一律取自 pi_con.plot_style，模組內不硬編
"""
