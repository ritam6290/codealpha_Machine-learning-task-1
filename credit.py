import os
import sys
import threading
import traceback
import zipfile
import tempfile

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, accuracy_score, classification_report
)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ----------------------------------------------------------------------
# Theme constants  (warm "slate & amber" palette)
# ----------------------------------------------------------------------
BG_DARK = "#13161c"
BG_PANEL = "#1b1f29"
BG_CARD = "#222733"
ACCENT = "#ffb347"
ACCENT_DARK = "#e0962a"
GOOD = "#5fd3a0"
BAD = "#ff6b6b"
TEXT_MAIN = "#f4f1ea"
TEXT_DIM = "#8c93a3"
BORDER = "#323a4a"
FONT_FAMILY = "Segoe UI" if sys.platform.startswith("win") else "Helvetica"

TARGET_COL = "SeriousDlqin2yrs"

# Default dataset path pre-filled in the GUI on launch.
DEFAULT_DATASET_PATH = r"C:\Users\ritam\Downloads\GiveMeSomeCredit.zip"

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

FRIENDLY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Credit Utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 Days Past Due",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines / Loans",
    "NumberOfTimes90DaysLate": "Times 90+ Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans / Lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 Days Past Due",
    "NumberOfDependents": "Number of Dependents",
}


# ----------------------------------------------------------------------
# Data + Modeling logic (kept separate from GUI code)
# ----------------------------------------------------------------------
class CreditModelPipeline:
    """Handles loading, feature engineering, training and prediction."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.model_name = None
        self.engineered_columns = None
        self.income_median = None
        self.util_cap = None
        self.debt_cap = None
        self.metrics = {}
        self.y_test = None
        self.probs = None
        self.preds = None
        self.feature_importance = None

    # -- loading -------------------------------------------------------
    @staticmethod
    def load_training_dataframe(path):
        """Accepts either a path to the zip (GiveMeSomeCredit.zip) or a
        direct path to a csv. Returns a raw dataframe with the target
        column present."""
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                csv_name = None
                for n in z.namelist():
                    if n.lower().endswith("cs-training.csv"):
                        csv_name = n
                        break
                if csv_name is None:
                    raise ValueError(
                        "Could not find 'cs-training.csv' inside the zip file."
                    )
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, index_col=0)
        elif path.lower().endswith(".csv"):
            df = pd.read_csv(path, index_col=0)
        else:
            raise ValueError("Please provide a .zip or .csv file path.")

        if TARGET_COL not in df.columns:
            raise ValueError(
                f"Expected target column '{TARGET_COL}' not found in the data."
            )
        return df

    # -- feature engineering --------------------------------------------
    def _engineer(self, df, fit=False):
        df = df.copy()

        if fit:
            self.income_median = df["MonthlyIncome"].median()
            self.util_cap = df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.99)
            self.debt_cap = df["DebtRatio"].quantile(0.99)

        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(self.income_median)
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

        df["RevolvingUtilizationOfUnsecuredLines"] = df[
            "RevolvingUtilizationOfUnsecuredLines"
        ].clip(upper=self.util_cap)
        df["DebtRatio"] = df["DebtRatio"].clip(upper=self.debt_cap)

        df["TotalPastDue"] = (
            df["NumberOfTime30-59DaysPastDueNotWorse"]
            + df["NumberOfTime60-89DaysPastDueNotWorse"]
            + df["NumberOfTimes90DaysLate"]
        )
        df["DebtPerLine"] = df["DebtRatio"] / (df["NumberOfOpenCreditLinesAndLoans"] + 1)
        df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

        if fit:
            self.engineered_columns = [c for c in df.columns if c != TARGET_COL]

        return df

    # -- training --------------------------------------------------------
    def train(self, df, model_choice, test_size=0.2, progress_cb=None):
        def report(pct, msg):
            if progress_cb:
                progress_cb(pct, msg)

        report(5, "Preparing data...")
        df = df.dropna(subset=[TARGET_COL])
        y = df[TARGET_COL].astype(int)
        X_raw = df.drop(columns=[TARGET_COL])

        report(15, "Engineering features...")
        X_eng_full = self._engineer(X_raw, fit=True)
        X_eng = X_eng_full[self.engineered_columns]

        report(25, "Splitting train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_eng, y, test_size=test_size, random_state=42, stratify=y
        )

        report(40, "Scaling features...")
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        report(55, f"Training {model_choice}...")
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_test_s)[:, 1]
            preds = model.predict(X_test_s)
            train_preds = model.predict(X_train_s)
            importance = np.abs(model.coef_[0])
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(
                max_depth=6, class_weight="balanced", random_state=42
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
            train_preds = model.predict(X_train)
            importance = model.feature_importances_
        else:  # Random Forest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
            train_preds = model.predict(X_train)
            importance = model.feature_importances_

        report(85, "Computing metrics...")
        self.model = model
        self.model_name = model_choice
        self.y_test = y_test
        self.probs = probs
        self.preds = preds
        self.feature_importance = dict(zip(self.engineered_columns, importance))

        self.metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "train_accuracy": accuracy_score(y_train, train_preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "pos_rate": float(y.mean()),
        }
        report(100, "Done.")
        return self.metrics

    # -- single applicant prediction -------------------------------------
    def predict_single(self, raw_values: dict):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        row = pd.DataFrame([raw_values])
        eng = self._engineer(row, fit=False)
        eng = eng[self.engineered_columns]

        if self.model_name == "Logistic Regression":
            X = self.scaler.transform(eng)
        else:
            X = eng

        prob = self.model.predict_proba(X)[0, 1]
        pred = int(prob >= 0.5)
        return pred, prob


# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class CreditScoringApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Credit Scoring Model — Creditworthiness Predictor")
        self.geometry("1180x780")
        self.minsize(1000, 680)
        self.configure(bg=BG_DARK)

        self.pipeline = CreditModelPipeline()
        self.df_loaded = None

        self._build_style()
        self._build_layout()

    # ------------------------------------------------------------------
    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG_DARK, foreground=TEXT_MAIN,
                         font=(FONT_FAMILY, 10))
        style.configure("TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("Card.TFrame", background=BG_CARD)

        style.configure("TLabel", background=BG_DARK, foreground=TEXT_MAIN,
                         font=(FONT_FAMILY, 10))
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=TEXT_MAIN)
        style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_MAIN)
        style.configure("Dim.TLabel", background=BG_DARK, foreground=TEXT_DIM)
        style.configure("PanelDim.TLabel", background=BG_PANEL, foreground=TEXT_DIM)
        style.configure("CardDim.TLabel", background=BG_CARD, foreground=TEXT_DIM)

        style.configure("Title.TLabel", background=BG_DARK, foreground=TEXT_MAIN,
                         font=(FONT_FAMILY, 20, "bold"))
        style.configure("Subtitle.TLabel", background=BG_DARK, foreground=TEXT_DIM,
                         font=(FONT_FAMILY, 10))
        style.configure("Section.TLabel", background=BG_PANEL, foreground=ACCENT,
                         font=(FONT_FAMILY, 12, "bold"))
        style.configure("MetricValue.TLabel", background=BG_CARD, foreground=ACCENT,
                         font=(FONT_FAMILY, 22, "bold"))
        style.configure("MetricLabel.TLabel", background=BG_CARD, foreground=TEXT_DIM,
                         font=(FONT_FAMILY, 9, "bold"))

        style.configure("TEntry", fieldbackground="#0f1217", foreground=TEXT_MAIN,
                         insertcolor=TEXT_MAIN, borderwidth=1, relief="flat")
        style.configure("TButton", background=ACCENT, foreground="#1b1f29",
                         font=(FONT_FAMILY, 10, "bold"), borderwidth=0, padding=10)
        style.map("TButton", background=[("active", ACCENT_DARK), ("disabled", "#3a3f4a")])

        style.configure("Secondary.TButton", background=BG_CARD, foreground=TEXT_MAIN,
                         font=(FONT_FAMILY, 10), borderwidth=1, padding=8)
        style.map("Secondary.TButton", background=[("active", "#2c3343")])

        style.configure("TCombobox", fieldbackground="#0f1217", background="#0f1217",
                         foreground=TEXT_MAIN, arrowcolor=ACCENT)

        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=TEXT_DIM,
                         padding=(18, 10), font=(FONT_FAMILY, 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", ACCENT)])

        style.configure("Horizontal.TProgressbar", background=ACCENT,
                         troughcolor=BG_CARD, borderwidth=0)

    # ------------------------------------------------------------------
    def _build_layout(self):
        header = ttk.Frame(self, style="TFrame")
        header.pack(fill="x", padx=24, pady=(22, 10))
        title_row = ttk.Frame(header, style="TFrame")
        title_row.pack(fill="x")
        ttk.Label(title_row, text="◆", style="Title.TLabel", foreground=ACCENT).pack(side="left", padx=(0, 10))
        ttk.Label(title_row, text="Credit Scoring Model", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="Predict creditworthiness from financial history — Logistic Regression, Decision Tree, or Random Forest",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        # ---- load panel ----
        load_panel = ttk.Frame(self, style="Panel.TFrame")
        load_panel.pack(fill="x", padx=24, pady=(0, 12))
        inner = ttk.Frame(load_panel, style="Panel.TFrame")
        inner.pack(fill="x", padx=18, pady=16)

        ttk.Label(inner, text="Dataset path (.zip or .csv)", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", columnspan=3, pady=(0, 6)
        )

        self.path_var = tk.StringVar(value=DEFAULT_DATASET_PATH)
        path_entry = ttk.Entry(inner, textvariable=self.path_var, width=70)
        path_entry.grid(row=1, column=0, sticky="we", padx=(0, 8))
        inner.columnconfigure(0, weight=1)

        ttk.Button(inner, text="Browse...", style="Secondary.TButton",
                   command=self._browse_file).grid(row=1, column=1, padx=4)

        self.model_var = tk.StringVar(value="Random Forest")
        model_combo = ttk.Combobox(
            inner, textvariable=self.model_var, state="readonly", width=20,
            values=["Logistic Regression", "Decision Tree", "Random Forest"],
        )
        model_combo.grid(row=1, column=2, padx=4)

        self.train_btn = ttk.Button(inner, text="Load && Train Model",
                                     command=self._on_train_clicked)
        self.train_btn.grid(row=1, column=3, padx=(8, 0))

        self.progress = ttk.Progressbar(inner, mode="determinate", length=200)
        self.progress.grid(row=2, column=0, columnspan=2, sticky="we", pady=(12, 0))
        self.status_label = ttk.Label(inner, text="Dataset path pre-filled below. Click Load && Train to begin.",
                                       style="PanelDim.TLabel")
        self.status_label.grid(row=2, column=2, columnspan=2, sticky="w", pady=(12, 0))

        # ---- tabs ----
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        self.tab_metrics = ttk.Frame(self.notebook, style="TFrame")
        self.tab_charts = ttk.Frame(self.notebook, style="TFrame")
        self.tab_predict = ttk.Frame(self.notebook, style="TFrame")
        self.tab_about = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.tab_metrics, text="  Metrics  ")
        self.notebook.add(self.tab_charts, text="  Charts  ")
        self.notebook.add(self.tab_predict, text="  Predict Applicant  ")
        self.notebook.add(self.tab_about, text="  About  ")

        self._build_metrics_tab()
        self._build_charts_tab()
        self._build_predict_tab()
        self._build_about_tab()

    # ------------------------------------------------------------------
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select GiveMeSomeCredit.zip or cs-training.csv",
            filetypes=[("Zip or CSV", "*.zip *.csv"), ("All files", "*.*")],
        )
        if path:
            self.path_var.set(path)

    # ------------------------------------------------------------------
    def _set_status(self, text, pct=None):
        self.status_label.config(text=text)
        if pct is not None:
            self.progress["value"] = pct
        self.update_idletasks()

    # ------------------------------------------------------------------
    def _on_train_clicked(self):
        path = self.path_var.get().strip().strip('"').strip("'")
        if not path:
            messagebox.showwarning("No path", "Please paste or browse to the dataset path first.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File not found:\n{path}")
            return

        self.train_btn.config(state="disabled")
        self.progress["value"] = 0
        self._set_status("Reading file...", 2)

        thread = threading.Thread(target=self._train_worker, args=(path,), daemon=True)
        thread.start()

    def _train_worker(self, path):
        try:
            df = CreditModelPipeline.load_training_dataframe(path)
            self.df_loaded = df
            model_choice = self.model_var.get()

            def progress_cb(pct, msg):
                self.after(0, lambda: self._set_status(msg, pct))

            metrics = self.pipeline.train(df, model_choice, progress_cb=progress_cb)
            self.after(0, lambda: self._on_train_success(metrics, model_choice))
        except Exception as e:
            tb = traceback.format_exc()
            self.after(0, lambda: self._on_train_error(str(e), tb))

    def _on_train_error(self, msg, tb):
        self.train_btn.config(state="normal")
        self._set_status(f"Error: {msg}", 0)
        messagebox.showerror("Training failed", f"{msg}\n\nDetails logged to console.")
        print(tb)

    def _on_train_success(self, metrics, model_choice):
        self.train_btn.config(state="normal")
        n = metrics["n_train"] + metrics["n_test"]
        self._set_status(
            f"✓ Trained {model_choice} on {n:,} records "
            f"({metrics['n_train']:,} train / {metrics['n_test']:,} test). "
            f"Base default rate: {metrics['pos_rate']*100:.1f}%",
            100,
        )
        self._update_metrics_tab(metrics, model_choice)
        self._update_charts_tab()
        self._populate_predict_defaults()

    # ------------------------------------------------------------------
    # METRICS TAB
    # ------------------------------------------------------------------
    def _build_metrics_tab(self):
        container = ttk.Frame(self.tab_metrics, style="TFrame")
        container.pack(fill="both", expand=True, padx=4, pady=4)

        self.metric_cards_frame = ttk.Frame(container, style="TFrame")
        self.metric_cards_frame.pack(fill="x", pady=(8, 16))

        self.metric_card_widgets = {}
        labels = [
            ("accuracy", "Test Accuracy"),
            ("train_accuracy", "Train Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-Score"),
            ("roc_auc", "ROC-AUC"),
        ]
        for i, (key, label) in enumerate(labels):
            card = ttk.Frame(self.metric_cards_frame, style="Card.TFrame")
            card.grid(row=0, column=i, padx=8, sticky="nsew")
            self.metric_cards_frame.columnconfigure(i, weight=1)
            val_lbl = ttk.Label(card, text="—", style="MetricValue.TLabel")
            val_lbl.pack(pady=(18, 2), padx=16)
            ttk.Label(card, text=label.upper(), style="MetricLabel.TLabel").pack(pady=(0, 16))
            self.metric_card_widgets[key] = val_lbl

        bottom = ttk.Frame(container, style="TFrame")
        bottom.pack(fill="both", expand=True)
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)

        # confusion matrix text panel
        cm_panel = ttk.Frame(bottom, style="Panel.TFrame")
        cm_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(cm_panel, text="Confusion Matrix", style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(14, 8)
        )
        self.cm_text = tk.Text(cm_panel, height=9, bg=BG_CARD, fg=TEXT_MAIN,
                                font=("Consolas" if sys.platform.startswith("win") else "Menlo", 11),
                                relief="flat", padx=14, pady=10, insertbackground=TEXT_MAIN,
                                highlightthickness=1, highlightbackground=BORDER)
        self.cm_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.cm_text.insert("1.0", "Train a model to see results here.")
        self.cm_text.config(state="disabled")

        # feature importance panel
        fi_panel = ttk.Frame(bottom, style="Panel.TFrame")
        fi_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(fi_panel, text="Top Feature Importance", style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(14, 8)
        )
        self.fi_text = tk.Text(fi_panel, height=9, bg=BG_CARD, fg=TEXT_MAIN,
                                font=("Consolas" if sys.platform.startswith("win") else "Menlo", 10),
                                relief="flat", padx=14, pady=10, insertbackground=TEXT_MAIN,
                                highlightthickness=1, highlightbackground=BORDER)
        self.fi_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.fi_text.insert("1.0", "Train a model to see results here.")
        self.fi_text.config(state="disabled")

    def _update_metrics_tab(self, metrics, model_choice):
        self.metric_card_widgets["accuracy"].config(text=f"{metrics['accuracy']*100:.1f}%")
        self.metric_card_widgets["train_accuracy"].config(text=f"{metrics['train_accuracy']*100:.1f}%")
        self.metric_card_widgets["precision"].config(text=f"{metrics['precision']*100:.1f}%")
        self.metric_card_widgets["recall"].config(text=f"{metrics['recall']*100:.1f}%")
        self.metric_card_widgets["f1"].config(text=f"{metrics['f1']:.3f}")
        self.metric_card_widgets["roc_auc"].config(text=f"{metrics['roc_auc']:.3f}")

        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        gap = metrics["train_accuracy"] - metrics["accuracy"]
        cm_str = (
            f"  Model: {model_choice}\n"
            f"  Test set size: {metrics['n_test']:,}\n"
            f"  Train accuracy: {metrics['train_accuracy']*100:.2f}%   "
            f"Test accuracy: {metrics['accuracy']*100:.2f}%   "
            f"(gap: {gap*100:+.2f}%)\n\n"
            f"                   Predicted\n"
            f"                Good      Default\n"
            f"  Actual Good   {tn:>6}     {fp:>6}\n"
            f"  Actual Default{fn:>6}     {tp:>6}\n\n"
            f"  True Negatives  (correctly approved): {tn:,}\n"
            f"  False Positives (wrongly flagged):    {fp:,}\n"
            f"  False Negatives (missed risk):        {fn:,}\n"
            f"  True Positives  (correctly flagged):  {tp:,}\n"
        )
        self.cm_text.config(state="normal")
        self.cm_text.delete("1.0", "end")
        self.cm_text.insert("1.0", cm_str)
        self.cm_text.config(state="disabled")

        fi = self.pipeline.feature_importance
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        max_val = max(v for _, v in sorted_fi) if sorted_fi else 1
        fi_lines = []
        for name, val in sorted_fi:
            display = FRIENDLY_NAMES.get(name, name)
            bar_len = int((val / max_val) * 20) if max_val else 0
            bar = "█" * bar_len
            fi_lines.append(f"  {display[:28]:<28} {bar:<20} {val:.4f}")
        self.fi_text.config(state="normal")
        self.fi_text.delete("1.0", "end")
        self.fi_text.insert("1.0", "\n".join(fi_lines))
        self.fi_text.config(state="disabled")

    # ------------------------------------------------------------------
    # CHARTS TAB
    # ------------------------------------------------------------------
    def _build_charts_tab(self):
        self.charts_frame = ttk.Frame(self.tab_charts, style="TFrame")
        self.charts_frame.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Label(
            self.charts_frame,
            text="Train a model to see the ROC curve and prediction distribution.",
            style="Dim.TLabel",
        ).pack(pady=40)

    def _update_charts_tab(self):
        for w in self.charts_frame.winfo_children():
            w.destroy()

        fig = Figure(figsize=(10.5, 4.6), dpi=100, facecolor=BG_DARK)

        # ROC curve
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_facecolor(BG_PANEL)
        fpr, tpr, _ = roc_curve(self.pipeline.y_test, self.pipeline.probs)
        auc = self.pipeline.metrics["roc_auc"]
        ax1.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
        ax1.plot([0, 1], [0, 1], color=TEXT_DIM, linestyle="--", linewidth=1)
        ax1.set_xlabel("False Positive Rate", color=TEXT_MAIN)
        ax1.set_ylabel("True Positive Rate", color=TEXT_MAIN)
        ax1.set_title("ROC Curve", color=TEXT_MAIN)
        ax1.tick_params(colors=TEXT_DIM)
        ax1.legend(loc="lower right", facecolor=BG_PANEL, edgecolor="none", labelcolor=TEXT_MAIN)
        for spine in ax1.spines.values():
            spine.set_color(TEXT_DIM)

        # Probability distribution by class
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_facecolor(BG_PANEL)
        probs = np.asarray(self.pipeline.probs)
        y_test = np.asarray(self.pipeline.y_test)
        ax2.hist(probs[y_test == 0], bins=30, alpha=0.7, color=GOOD, label="Good (actual)")
        ax2.hist(probs[y_test == 1], bins=30, alpha=0.7, color=BAD, label="Default (actual)")
        ax2.set_xlabel("Predicted Risk Probability", color=TEXT_MAIN)
        ax2.set_ylabel("Count", color=TEXT_MAIN)
        ax2.set_title("Predicted Risk Distribution", color=TEXT_MAIN)
        ax2.tick_params(colors=TEXT_DIM)
        ax2.legend(loc="upper right", facecolor=BG_PANEL, edgecolor="none", labelcolor=TEXT_MAIN)
        for spine in ax2.spines.values():
            spine.set_color(TEXT_DIM)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # PREDICT TAB
    # ------------------------------------------------------------------
    def _build_predict_tab(self):
        outer = ttk.Frame(self.tab_predict, style="TFrame")
        outer.pack(fill="both", expand=True, padx=4, pady=4)

        left = ttk.Frame(outer, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(left, text="Applicant Financial Profile", style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(14, 10)
        )

        form = ttk.Frame(left, style="Panel.TFrame")
        form.pack(fill="x", padx=16)

        self.predict_vars = {}
        defaults = {
            "RevolvingUtilizationOfUnsecuredLines": "0.3",
            "age": "40",
            "NumberOfTime30-59DaysPastDueNotWorse": "0",
            "DebtRatio": "0.3",
            "MonthlyIncome": "5000",
            "NumberOfOpenCreditLinesAndLoans": "6",
            "NumberOfTimes90DaysLate": "0",
            "NumberRealEstateLoansOrLines": "1",
            "NumberOfTime60-89DaysPastDueNotWorse": "0",
            "NumberOfDependents": "0",
        }
        for i, feat in enumerate(RAW_FEATURES):
            ttk.Label(form, text=FRIENDLY_NAMES[feat], style="Panel.TLabel").grid(
                row=i, column=0, sticky="w", pady=6, padx=(0, 10)
            )
            var = tk.StringVar(value=defaults[feat])
            entry = ttk.Entry(form, textvariable=var, width=18)
            entry.grid(row=i, column=1, sticky="e", pady=6)
            self.predict_vars[feat] = var
        form.columnconfigure(1, weight=1)

        self.predict_btn = ttk.Button(
            left, text="Predict Creditworthiness", command=self._on_predict_clicked
        )
        self.predict_btn.pack(anchor="w", padx=16, pady=16)

        # right: result panel
        right = ttk.Frame(outer, style="Panel.TFrame")
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        ttk.Label(right, text="Prediction Result", style="Section.TLabel").pack(
            anchor="w", padx=16, pady=(14, 10)
        )

        self.result_card = ttk.Frame(right, style="Card.TFrame")
        self.result_card.pack(fill="x", padx=16, pady=8)
        self.result_title = ttk.Label(self.result_card, text="No prediction yet",
                                       style="Card.TLabel", font=(FONT_FAMILY, 16, "bold"))
        self.result_title.pack(pady=(20, 4), padx=20)
        self.result_prob = ttk.Label(self.result_card, text="Train a model and click Predict.",
                                      style="CardDim.TLabel")
        self.result_prob.pack(pady=(0, 20), padx=20)

        self.result_bar_canvas = tk.Canvas(right, height=28, bg=BG_PANEL, highlightthickness=0)
        self.result_bar_canvas.pack(fill="x", padx=16, pady=(4, 16))

        ttk.Label(
            right,
            text="The model estimates probability of serious delinquency\n"
                 "(90+ days past due, bankruptcy, etc.) within 2 years.\n"
                 "Threshold for 'High Risk' classification is 50%.",
            style="PanelDim.TLabel", justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _populate_predict_defaults(self):
        # No-op placeholder in case we want to seed from data medians later
        pass

    def _on_predict_clicked(self):
        if self.pipeline.model is None:
            messagebox.showwarning("No model", "Please load data and train a model first.")
            return
        try:
            raw_values = {}
            for feat in RAW_FEATURES:
                val = self.predict_vars[feat].get().strip()
                raw_values[feat] = float(val)
        except ValueError:
            messagebox.showerror("Invalid input", "Please make sure all fields contain valid numbers.")
            return

        try:
            pred, prob = self.pipeline.predict_single(raw_values)
        except Exception as e:
            messagebox.showerror("Prediction failed", str(e))
            return

        if pred == 1:
            self.result_title.config(text="⚠ High Risk", foreground=BAD)
            self.result_prob.config(text=f"Estimated default probability: {prob*100:.1f}%")
        else:
            self.result_title.config(text="✓ Creditworthy", foreground=GOOD)
            self.result_prob.config(text=f"Estimated default probability: {prob*100:.1f}%")

        self.result_bar_canvas.delete("all")
        w = self.result_bar_canvas.winfo_width() or 400
        h = 28
        self.result_bar_canvas.create_rectangle(0, 0, w, h, fill=BG_CARD, outline="")
        bar_w = int(w * min(max(prob, 0), 1))
        color = BAD if prob >= 0.5 else GOOD
        self.result_bar_canvas.create_rectangle(0, 0, bar_w, h, fill=color, outline="")
        self.result_bar_canvas.create_line(w * 0.5, 0, w * 0.5, h, fill=TEXT_DIM, dash=(2, 2))

    # ------------------------------------------------------------------
    # ABOUT TAB
    # ------------------------------------------------------------------
    def _build_about_tab(self):
        panel = ttk.Frame(self.tab_about, style="Panel.TFrame")
        panel.pack(fill="both", expand=True, padx=4, pady=4)
        text = (
            "Credit Scoring Model\n\n"
            "Objective: Predict an individual's creditworthiness using past financial data.\n\n"
            "Approach: Classification algorithms — Logistic Regression, Decision Tree, Random Forest.\n\n"
            "Feature engineering:\n"
            "  • Missing MonthlyIncome imputed with median; missing NumberOfDependents filled with 0.\n"
            "  • Outlier capping at the 99th percentile for utilization and debt ratio.\n"
            "  • Derived features: TotalPastDue, DebtPerLine, IncomePerDependent.\n"
            "  • Class imbalance handled via class_weight='balanced'.\n\n"
            "Evaluation metrics: Train Accuracy, Test Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.\n\n"
            "Dataset: 'Give Me Some Credit' — income, debts, payment history, credit lines, "
            "delinquency history, dependents.\n\n"
            "Usage:\n"
            "  1. The dataset path is pre-filled (edit it if your file lives elsewhere).\n"
            "  2. Choose a model and click 'Load & Train Model'.\n"
            "  3. Review metrics and charts in their respective tabs.\n"
            "  4. Use the 'Predict Applicant' tab to score a new individual.\n"
        )
        lbl = tk.Text(panel, bg=BG_PANEL, fg=TEXT_MAIN, relief="flat",
                       font=(FONT_FAMILY, 11), wrap="word", padx=20, pady=20)
        lbl.pack(fill="both", expand=True)
        lbl.insert("1.0", text)
        lbl.config(state="disabled")


if __name__ == "__main__":
    app = CreditScoringApp()
    app.mainloop()
