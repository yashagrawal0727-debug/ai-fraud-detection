import datetime
import pickle

import customtkinter as ctk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")

# =========================
# APP CONFIG
# =========================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PURPLE_DARK = "#2D0A6B"
PURPLE_MID = "#5B21B6"
PURPLE_LIGHT = "#7C3AED"
PURPLE_ACCENT = "#8B5CF6"
WHITE = "#FFFFFF"
WHITE_80 = "#CCCCFF"
INPUT_BG = "#6D28D9"
ERROR_RED = "#EF4444"
SUCCESS_GREEN = "#22C55E"

app = ctk.CTk()
app.title("AI Fraud Detection")
app.geometry("1100x900")
app.configure(fg_color=PURPLE_DARK)

# =========================
# LOAD MODEL
# =========================

try:
    model = pickle.load(open("fraud_model.pkl", "rb"))
except FileNotFoundError:
    model = None

fraud_count = 0
genuine_count = 0
last_probability = 0.0

# =========================
# SCROLLABLE CANVAS
# =========================

outer_frame = ctk.CTkFrame(app, fg_color=PURPLE_DARK)
outer_frame.pack(fill="both", expand=True)

root_canvas = ctk.CTkCanvas(outer_frame, bg=PURPLE_DARK, highlightthickness=0)
root_canvas.pack(side="left", fill="both", expand=True)

scrollbar = ctk.CTkScrollbar(
    outer_frame, orientation="vertical", command=root_canvas.yview
)
scrollbar.pack(side="right", fill="y")
root_canvas.configure(yscrollcommand=scrollbar.set)

scroll_frame = ctk.CTkFrame(root_canvas, fg_color=PURPLE_DARK)
scroll_window = root_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")


def _on_frame_configure(event):
    root_canvas.configure(scrollregion=root_canvas.bbox("all"))


def _on_canvas_configure(event):
    root_canvas.itemconfig(scroll_window, width=event.width)


scroll_frame.bind("<Configure>", _on_frame_configure)
root_canvas.bind("<Configure>", _on_canvas_configure)


def _mousewheel(event):
    root_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


app.bind_all("<MouseWheel>", _mousewheel)
app.bind_all("<Button-4>", lambda e: root_canvas.yview_scroll(-1, "units"))
app.bind_all("<Button-5>", lambda e: root_canvas.yview_scroll(1, "units"))

# =========================
# HERO
# =========================

hero = ctk.CTkFrame(scroll_frame, fg_color=PURPLE_DARK)
hero.pack(fill="x", pady=(30, 10))

ctk.CTkLabel(
    hero, text="Detect. Prevent. Secure.", font=("Arial", 38, "bold"), text_color=WHITE
).pack()
ctk.CTkLabel(
    hero,
    text="Next-Gen AI protecting transactions across the world",
    font=("Arial", 16),
    text_color=WHITE_80,
).pack(pady=(4, 0))

# =========================
# HELPERS
# =========================

ENTRY_W = 460
ENTRY_H = 44
SMALL_W = 100


def entry_style():
    return dict(
        width=ENTRY_W,
        height=ENTRY_H,
        fg_color=INPUT_BG,
        border_color=PURPLE_ACCENT,
        border_width=1,
        text_color=WHITE,
        placeholder_text_color=WHITE_80,
        corner_radius=10,
    )


def small_entry_style(w=SMALL_W):
    s = entry_style()
    s["width"] = w
    return s


def make_label(parent, text):
    ctk.CTkLabel(
        parent, text=text, font=("Arial", 13), text_color=WHITE_80, anchor="w"
    ).pack(fill="x", padx=40, pady=(8, 2))


def make_error_label(parent):
    lbl = ctk.CTkLabel(
        parent, text="", font=("Arial", 11), text_color=ERROR_RED, anchor="w"
    )
    lbl.pack(fill="x", padx=40)
    return lbl


def set_error(entry, err_lbl, msg):
    entry.configure(border_color=ERROR_RED)
    err_lbl.configure(text=msg)


def clear_error(entry, err_lbl):
    entry.configure(border_color=PURPLE_ACCENT)
    err_lbl.configure(text="")


# =========================
# FORM CARD
# =========================

form_card = ctk.CTkFrame(scroll_frame, fg_color=PURPLE_MID, corner_radius=20)
form_card.pack(padx=200, pady=(20, 10), fill="x")

ctk.CTkLabel(
    form_card,
    text="Enter Transaction Details",
    font=("Arial", 22, "bold"),
    text_color=WHITE,
).pack(pady=(24, 16))

# ── Transaction Type ──
make_label(form_card, "Transaction Type")
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
type_menu = ctk.CTkOptionMenu(
    form_card,
    values=TRANSACTION_TYPES,
    width=ENTRY_W,
    height=ENTRY_H,
    fg_color=INPUT_BG,
    button_color=PURPLE_ACCENT,
    button_hover_color=PURPLE_LIGHT,
    text_color=WHITE,
    corner_radius=10,
)
type_menu.pack()
type_menu.set("PAYMENT")

# ── Amount ──
make_label(form_card, "Transaction Amount (₹)")
amount_entry = ctk.CTkEntry(form_card, placeholder_text="Enter amount", **entry_style())
amount_entry.pack()
amount_err = make_error_label(form_card)

# ── Date ──
make_label(form_card, "Transaction Date")
date_row = ctk.CTkFrame(form_card, fg_color="transparent")
date_row.pack()
dd_entry = ctk.CTkEntry(date_row, placeholder_text="DD", **small_entry_style(90))
dd_entry.pack(side="left", padx=4)
mm_entry = ctk.CTkEntry(date_row, placeholder_text="MM", **small_entry_style(90))
mm_entry.pack(side="left", padx=4)
yyyy_entry = ctk.CTkEntry(date_row, placeholder_text="YYYY", **small_entry_style(130))
yyyy_entry.pack(side="left", padx=4)

date_err_row = ctk.CTkFrame(form_card, fg_color="transparent")
date_err_row.pack(fill="x", padx=40)
date_err = ctk.CTkLabel(
    date_err_row, text="", font=("Arial", 11), text_color=ERROR_RED, anchor="w"
)
date_err.pack(fill="x")


def clear_date_error():
    for e in (dd_entry, mm_entry, yyyy_entry):
        e.configure(border_color=PURPLE_ACCENT)
    date_err.configure(text="")


def set_date_error(msg):
    for e in (dd_entry, mm_entry, yyyy_entry):
        e.configure(border_color=ERROR_RED)
    date_err.configure(text=msg)


# ── Time ──
make_label(form_card, "Transaction Time")
time_row = ctk.CTkFrame(form_card, fg_color="transparent")
time_row.pack()
hours_entry = ctk.CTkEntry(time_row, placeholder_text="HH", **small_entry_style(85))
hours_entry.pack(side="left", padx=4)
minutes_entry = ctk.CTkEntry(time_row, placeholder_text="MM", **small_entry_style(85))
minutes_entry.pack(side="left", padx=4)
seconds_entry = ctk.CTkEntry(time_row, placeholder_text="SS", **small_entry_style(85))
seconds_entry.pack(side="left", padx=4)
ampm_menu = ctk.CTkOptionMenu(
    time_row,
    values=["AM", "PM"],
    width=80,
    height=ENTRY_H,
    fg_color=INPUT_BG,
    button_color=PURPLE_ACCENT,
    button_hover_color=PURPLE_LIGHT,
    text_color=WHITE,
    corner_radius=10,
)
ampm_menu.pack(side="left", padx=4)
time_err = make_error_label(form_card)

# ── Location ──
make_label(form_card, "Location (City)")
location_entry = ctk.CTkEntry(
    form_card, placeholder_text="Enter your city", **entry_style()
)
location_entry.pack()
loc_err = make_error_label(form_card)

# ── Result ──
result_label = ctk.CTkLabel(
    form_card, text="", font=("Arial", 20, "bold"), text_color=WHITE
)
result_label.pack(pady=(14, 2))
probability_label = ctk.CTkLabel(
    form_card, text="", font=("Arial", 14), text_color=WHITE_80
)
probability_label.pack()
city_label = ctk.CTkLabel(form_card, text="", font=("Arial", 14), text_color=WHITE_80)
city_label.pack(pady=(0, 8))

# ── Submit ──
submit_button = ctk.CTkButton(
    form_card,
    text="Check Fraud",
    height=46,
    width=ENTRY_W,
    fg_color=WHITE,
    hover_color="#E8E8FF",
    text_color=PURPLE_DARK,
    font=("Arial", 16, "bold"),
    corner_radius=10,
    command=lambda: check_fraud(),
)
submit_button.pack(pady=(6, 28))

# =========================
# ANALYTICS
# =========================

ctk.CTkLabel(
    scroll_frame,
    text="Fraud Detection Analytics",
    font=("Arial", 26, "bold"),
    text_color=WHITE,
).pack(pady=(30, 16))

stats_row = ctk.CTkFrame(scroll_frame, fg_color=PURPLE_DARK)
stats_row.pack(padx=100, fill="x")
for c in range(3):
    stats_row.columnconfigure(c, weight=1)


def make_stat_card(parent, col, title):
    card = ctk.CTkFrame(parent, fg_color=PURPLE_MID, corner_radius=16)
    card.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")
    ctk.CTkLabel(card, text=title, font=("Arial", 14, "bold"), text_color=WHITE).pack(
        pady=(16, 4)
    )
    lbl = ctk.CTkLabel(card, text="0", font=("Arial", 28, "bold"), text_color=WHITE)
    lbl.pack(pady=(0, 16))
    return lbl


total_label = make_stat_card(stats_row, 0, "Total\nTransactions")
fraud_label = make_stat_card(stats_row, 1, "Fraud Detected")
genuine_label = make_stat_card(stats_row, 2, "Genuine")

legend_frame = ctk.CTkFrame(scroll_frame, fg_color=PURPLE_DARK)
legend_frame.pack(pady=(8, 4))
for color, lbl_text in [(ERROR_RED, "Fraud"), (SUCCESS_GREEN, "Genuine")]:
    dot = ctk.CTkFrame(
        legend_frame, fg_color=color, width=14, height=14, corner_radius=7
    )
    dot.pack(side="left", padx=(10, 4))
    ctk.CTkLabel(
        legend_frame, text=lbl_text, font=("Arial", 13), text_color=WHITE
    ).pack(side="left", padx=(0, 10))

charts_card = ctk.CTkFrame(scroll_frame, fg_color=PURPLE_MID, corner_radius=16)
charts_card.pack(padx=100, pady=(10, 30), fill="x")
charts_inner = ctk.CTkFrame(charts_card, fg_color="transparent")
charts_inner.pack(fill="both", expand=True, padx=20, pady=20)

bar_frame = ctk.CTkFrame(charts_inner, fg_color="transparent")
bar_frame.pack(side="left", fill="both", expand=True)
pie_frame = ctk.CTkFrame(charts_inner, fg_color="transparent")
pie_frame.pack(side="left", fill="both", expand=True)

bar_canvas_widget = None
pie_canvas_widget = None
CHART_BG = "#5B21B6"


def update_charts(fraud_c, genuine_c, probability):
    global bar_canvas_widget, pie_canvas_widget

    if bar_canvas_widget:
        bar_canvas_widget.get_tk_widget().destroy()
    fig_bar, ax_bar = plt.subplots(figsize=(4.5, 3.5))
    fig_bar.patch.set_facecolor(CHART_BG)
    ax_bar.set_facecolor(CHART_BG)
    values = [fraud_c, genuine_c]
    bars = ax_bar.bar(
        ["Fraud", "Genuine"],
        values,
        color=[ERROR_RED, SUCCESS_GREEN],
        width=0.4,
        zorder=2,
    )
    for bar, val in zip(bars, values):
        if val > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(val),
                ha="center",
                va="bottom",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
    ax_bar.set_ylim(0, max(max(values) + 2, 5))
    ax_bar.set_title("Transaction Summary", color="white", fontsize=13, pad=10)
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor(PURPLE_LIGHT)
    ax_bar.yaxis.set_tick_params(labelcolor="white")
    ax_bar.xaxis.set_tick_params(labelcolor="white")
    bar_canvas_widget = FigureCanvasTkAgg(fig_bar, master=bar_frame)
    bar_canvas_widget.draw()
    bar_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig_bar)

    if pie_canvas_widget:
        pie_canvas_widget.get_tk_widget().destroy()
    fig_pie, ax_pie = plt.subplots(figsize=(4.5, 3.5))
    fig_pie.patch.set_facecolor(CHART_BG)
    ax_pie.set_facecolor(CHART_BG)
    fraud_pct = probability * 100
    genuine_pct = (1 - probability) * 100
    wedges, texts, autotexts = ax_pie.pie(
        [fraud_pct, genuine_pct],
        labels=["Risk", "Genuine"],
        colors=[ERROR_RED, SUCCESS_GREEN],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": CHART_BG, "linewidth": 2},
    )
    for t in texts:
        t.set_color("white")
        t.set_fontsize(12)
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax_pie.set_title("Fraud Risk %", color="white", fontsize=13, pad=10)
    pie_canvas_widget = FigureCanvasTkAgg(fig_pie, master=pie_frame)
    pie_canvas_widget.draw()
    pie_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
    plt.close(fig_pie)


update_charts(0, 0, 0.0)

# =========================
# VALIDATION
# =========================


def validate_form():
    ok = True

    # Amount
    raw_amt = amount_entry.get().strip()
    if not raw_amt:
        set_error(amount_entry, amount_err, "Amount is required.")
        ok = False
    else:
        try:
            amt = float(raw_amt)
            if amt <= 0:
                raise ValueError
            clear_error(amount_entry, amount_err)
        except ValueError:
            set_error(amount_entry, amount_err, "Enter a valid positive number.")
            ok = False

    # Date
    dd_raw, mm_raw, yyyy_raw = (
        dd_entry.get().strip(),
        mm_entry.get().strip(),
        yyyy_entry.get().strip(),
    )
    if not dd_raw or not mm_raw or not yyyy_raw:
        set_date_error("Day, month and year are all required.")
        ok = False
    else:
        try:
            datetime.date(int(yyyy_raw), int(mm_raw), int(dd_raw))
            clear_date_error()
        except ValueError:
            set_date_error("Enter a valid date (DD 1-31, MM 1-12, YYYY 4 digits).")
            ok = False

    # Time
    h_raw = hours_entry.get().strip()
    if not h_raw:
        set_error(hours_entry, time_err, "Hour is required.")
        ok = False
    else:
        try:
            hh = int(h_raw)
            if not (1 <= hh <= 12):
                raise ValueError
            mm_t = (
                int(minutes_entry.get().strip()) if minutes_entry.get().strip() else 0
            )
            ss_t = (
                int(seconds_entry.get().strip()) if seconds_entry.get().strip() else 0
            )
            if not (0 <= mm_t <= 59) or not (0 <= ss_t <= 59):
                raise ValueError
            clear_error(hours_entry, time_err)
        except ValueError:
            set_error(hours_entry, time_err, "HH 1-12, MM 0-59, SS 0-59.")
            ok = False

    # Location
    raw_loc = location_entry.get().strip()
    if not raw_loc:
        set_error(location_entry, loc_err, "City is required.")
        ok = False
    elif not raw_loc.replace(" ", "").isalpha():
        set_error(location_entry, loc_err, "City should contain only letters.")
        ok = False
    else:
        clear_error(location_entry, loc_err)

    if not ok:
        return False, {}

    hh = int(hours_entry.get().strip())
    if ampm_menu.get() == "PM" and hh != 12:
        hh += 12
    elif ampm_menu.get() == "AM" and hh == 12:
        hh = 0

    return True, {
        "amount": float(amount_entry.get().strip()),
        "hours": hh,
        "location": location_entry.get().strip(),
        "txn_type": type_menu.get(),
    }


# =========================
# PREDICTION
# =========================

# CASH_IN is dropped baseline — all zeros = CASH_IN
TYPE_COLUMNS = ["type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

# Population average txn_freq per type from PaySim (model was trained on this scale)
FREQ_PRIOR = {
    "PAYMENT": 3.2,
    "TRANSFER": 1.8,
    "CASH_OUT": 2.1,
    "DEBIT": 2.4,
    "CASH_IN": 2.0,
}


def check_fraud():
    global fraud_count, genuine_count, last_probability

    valid, parsed = validate_form()
    if not valid:
        result_label.configure(
            text="⚠  Please fix the errors above.", text_color="orange"
        )
        return

    try:
        amount = parsed["amount"]
        hours = parsed["hours"]
        location = parsed["location"]
        txn_type = parsed["txn_type"]

        log_amount = np.log1p(amount)
        time_sin = np.sin(2 * np.pi * hours / 24)
        time_cos = np.cos(2 * np.pi * hours / 24)
        txn_freq = FREQ_PRIOR.get(txn_type, 2.0)

        type_values = {col: 0 for col in TYPE_COLUMNS}
        col_name = f"type_{txn_type}"
        if col_name in type_values:
            type_values[col_name] = 1

        features = pd.DataFrame(
            [
                {
                    "log_amount": log_amount,
                    "time_sin": time_sin,
                    "time_cos": time_cos,
                    "txn_freq": txn_freq,
                    **type_values,
                }
            ]
        )

        if model is None:
            raise RuntimeError(
                "fraud_model.pkl not found — place it next to this script."
            )

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        last_probability = probability

        if prediction == 1:
            fraud_count += 1
            result_label.configure(text="⚠  Fraud Detected", text_color=ERROR_RED)
        else:
            genuine_count += 1
            result_label.configure(
                text="✔  Genuine Transaction", text_color=SUCCESS_GREEN
            )

        probability_label.configure(text=f"Fraud Probability: {probability * 100:.2f}%")
        city_label.configure(text=f"City: {location.title()}")

        total = fraud_count + genuine_count
        total_label.configure(text=str(total))
        fraud_label.configure(text=str(fraud_count))
        genuine_label.configure(text=str(genuine_count))

        update_charts(fraud_count, genuine_count, probability)

    except Exception as e:
        result_label.configure(text=f"Error: {e}", text_color="orange")


# =========================
# RUN
# =========================

app.mainloop()
