# -*- coding: utf-8 -*-
"""
CARBSE Heatmap Generator (Type-1, single city)
Flask + Plotly (responsive) + Data preview + Custom colorscale
- Date format radios wired to x-axis tick labels
- Scale max defaults: DBT->50, RH->100 (UI + backend)
- Data tab alignment fix
- Download plot as JPG (via kaleido)
"""
import os
import uuid
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="Templates")
app.secret_key = "change-this"  # required for flash()

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

# session_id -> {"path":..., "file":..., "columns":[...], "data_html":"<table>...</table>"}
sessions = {}

# Map UI color names to hex (case-insensitive)
COLOR_MAP = {
    "white":   "#ffffff",
    "navyblue":"#001f3f",
    "blue":    "#0000ff",
    "cyan":    "#00ffff",
    "yellow":  "#ffff00",
    "red":     "#ff0000",
    "darkred": "#8b0000",
    "black":   "#000000",
    "grey":    "#808080",
}

# Map date-format radio values -> Plotly/D3 tickformat strings
DATE_FMT_MAP = {
    "Mon-YY":     "%b-%y",    # Jan-17
    "01-Jan":     "%d-%b",    # 01-Jan
    "01-01-17":   "%d-%m-%y", # 01-01-17
    "01-January": "%d-%B",    # 01-January
    "Mon":        "%a",       # Mon
    "Monday":     "%A",       # Monday
    "Jan":        "%b",       # Jan
    "01":         "%m",       # 01 (month number)
    "17":         "%y",       # 2-digit year
    "2017":       "%Y",       # 4-digit year
}

def clean_preview(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed index cols and reorder to keep Timestamp/DBT/RH first."""
    # Drop any "Unnamed: ..." columns (often CSV index artifacts)
    mask = ~df.columns.astype(str).str.match(r"^Unnamed")
    df = df.loc[:, mask]
    # Reorder if the main cols exist
    order = [c for c in ["Timestamp", "DBT", "RH"] if c in df.columns]
    rest  = [c for c in df.columns if c not in order]
    return df[order + rest]

def df_to_html_table(df, max_rows=200):
    """Bootstrap table for Data tab (limit rows to keep it light)."""
    if df is None:
        return None
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    df = clean_preview(df)
    return df.to_html(classes="table table-sm table-striped table-hover",
                      index=False, border=0, na_rep="")

def build_colorscale(selected):
    """
    Build a Plotly colorscale from the checkbox order.
    If fewer than 2 colors are selected, return 'Viridis'.
    """
    if not selected or len(selected) < 2:
        return "Viridis"
    hexes = []
    for name in selected:
        key = (name or "").strip().lower()
        hexes.append(COLOR_MAP.get(key, name))  # fall back to raw string if unknown
    n = len(hexes)
    return [[i/(n-1), hexes[i]] for i in range(n)]

@app.route("/", methods=["GET"])
def root():
    return render_template("heatmap_app_00.html", form={}, chosen_colors=[], data_html=None)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/upload", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return redirect(url_for("root"))

    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename.strip():
        flash("No selected file. Please choose a CSV.")
        return redirect(url_for("root"))

    sid = str(uuid.uuid4())
    session_path = os.path.join(UPLOAD_DIR, sid)
    os.makedirs(session_path, exist_ok=True)

    file_path = os.path.join(session_path, secure_filename(uploaded.filename))
    uploaded.save(file_path)

    try:
        df_head = pd.read_csv(file_path, nrows=5)
        cols = df_head.columns.tolist()
        df_preview = pd.read_csv(file_path, nrows=200)
        data_html = df_to_html_table(df_preview, max_rows=200)
    except Exception as e:
        flash(f"Failed to read CSV: {e}")
        return redirect(url_for("root"))

    sessions[sid] = {"path": session_path, "file": file_path, "columns": cols, "data_html": data_html}

    return render_template(
        "heatmap_app_00.html",
        session_id=sid,
        columns=cols,
        form={},
        chosen_colors=[],
        data_html=data_html
    )

@app.route("/generate-heatmap", methods=["POST"])
def generate_heatmap():
    form_state = request.form.to_dict()
    chosen_colors = request.form.getlist("colors[]")

    sid = form_state.get("session_id", "").strip()
    if not sid or sid not in sessions:
        flash("Please upload a file first.")
        return redirect(url_for("root"))

    sess = sessions[sid]
    file_path = sess["file"]

    value_column = (form_state.get("parameter") or "").strip()
    if not value_column:
        flash("Please select a parameter (DBT or RH).")
        return render_template("heatmap_app_00.html",
                               session_id=sid, columns=sess.get("columns"),
                               form=form_state, chosen_colors=chosen_colors,
                               data_html=sess.get("data_html"))

    # Date format selection
    date_format_choice = (form_state.get("date_format") or "Mon-YY").strip()
    tickformat = DATE_FMT_MAP.get(date_format_choice, "%b-%y")

    chart_title  = (form_state.get("chart_title") or "").strip()
    x_label      = (form_state.get("x_label") or "Date").strip()
    y_label      = (form_state.get("y_label") or "Hour").strip()
    legend_label = (form_state.get("legend_label") or value_column).strip()

    def to_int(val, default):
        try: return int(val)
        except: return default
    image_width  = to_int(form_state.get("image_width") or form_state.get("img_width"), 1200)
    image_height = to_int(form_state.get("image_height") or form_state.get("img_height"), 600)

    upper_time       = (form_state.get("upper_limit_time") or "").strip()
    lower_time       = (form_state.get("lower_limit_time") or "").strip()
    upper_line_style = (form_state.get("upper_line_style") or "dotted").strip()
    lower_line_style = (form_state.get("lower_line_style") or "dotted").strip()
    upper_line_color = (form_state.get("upper_line_color") or "black").strip()
    lower_line_color = (form_state.get("lower_line_color") or "black").strip()

    def to_float_or_none(x):
        try: return float(x)
        except: return None

    # Dynamic default for scale max based on parameter
    default_max = 50.0 if value_column == "DBT" else (100.0 if value_column == "RH" else None)
    zmin = to_float_or_none(form_state.get("scale_min"))
    zmax = to_float_or_none(form_state.get("scale_max"))
    if zmax is None and default_max is not None:
        zmax = default_max
        form_state["scale_max"] = str(int(default_max))  # reflect in UI after render

    # read + data preview (first rows)
    try:
        df = pd.read_csv(file_path)
        data_html = df_to_html_table(df, max_rows=200)
    except Exception as e:
        flash(f"Failed to read CSV after upload: {e}")
        return render_template("heatmap_app_00.html",
                               session_id=sid, columns=sess.get("columns"),
                               form=form_state, chosen_colors=chosen_colors,
                               data_html=sess.get("data_html"))

    # Robust Timestamp parse & drop NaT to avoid 1970 issue
    if "Timestamp" not in df.columns:
        flash("CSV must include a 'Timestamp' column.")
        return render_template("heatmap_app_00.html",
                               session_id=sid, columns=sess.get("columns"),
                               form=form_state, chosen_colors=chosen_colors,
                               data_html=data_html)

    ts_all = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    mask = ts_all.notna()
    df = df.loc[mask].copy()
    ts = ts_all.loc[mask]
    df["__Date__"] = ts.dt.normalize()
    df["__Hour__"] = ts.dt.hour

    if value_column not in df.columns:
        flash(f"CSV does not contain column: {value_column}")
        return render_template("heatmap_app_00.html",
                               session_id=sid, columns=sess.get("columns"),
                               form=form_state, chosen_colors=chosen_colors,
                               data_html=data_html)

    # Pivot for heatmap
    pivot_df = df.pivot_table(index="__Hour__", columns="__Date__", values=value_column, aggfunc="mean")
    pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)

    x_vals = list(pivot_df.columns)  # pandas Timestamps (normalized dates)
    y_vals = list(pivot_df.index)

    # Colorscale from selected checkboxes
    colorscale = build_colorscale(chosen_colors)

    heat_kwargs = {
        "z": pivot_df.values,
        "x": x_vals,
        "y": y_vals,
        "colorscale": colorscale,
        "colorbar": dict(title=legend_label)
    }
    if zmin is not None: heat_kwargs["zmin"] = zmin
    if zmax is not None: heat_kwargs["zmax"] = zmax

    fig = go.Figure(data=go.Heatmap(**heat_kwargs))
    fig.update_layout(
        title=chart_title or f"Heatmap ({value_column})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=image_height,
        margin=dict(l=40, r=40, t=80, b=80),
    )

    # X-axis: date type, clamp range, and apply tickformat from radios
    if x_vals:
        fig.update_xaxes(type="date",
                         range=[x_vals[0], x_vals[-1]],
                         tickformat=tickformat)

    # Band lines at selected hours
    def dash_style(name: str) -> str:
        n = (name or "").lower()
        if n in ("dotted", "dot", "dots"): return "dot"
        if n in ("dash", "dashed"):        return "dash"
        return "solid"

    def to_hour(s: str):
        try: return int(s.split(":")[0]) if s else None
        except: return None

    uh, lh = to_hour(upper_time), to_hour(lower_time)
    if uh is not None and x_vals:
        fig.add_shape(type="line", x0=x_vals[0], x1=x_vals[-1], y0=uh, y1=uh,
                      xref="x", yref="y",
                      line=dict(color=upper_line_color, dash=dash_style(upper_line_style)))
    if lh is not None and x_vals:
        fig.add_shape(type="line", x0=x_vals[0], x1=x_vals[-1], y0=lh, y1=lh,
                      xref="x", yref="y",
                      line=dict(color=lower_line_color, dash=dash_style(lower_line_style)))

    # HTML (interactive) and JPG (static) outputs
    graph_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={"responsive": True})

    # Save JPG for download
    img_path = os.path.join(PLOT_DIR, f"{sid}.jpg")
    try:
        fig.write_image(img_path, format="jpg", width=image_width, height=image_height, scale=2)
    except Exception as e:
        # If kaleido not installed, inform but still render HTML plot
        flash("Static image export requires 'kaleido'. Run: pip install kaleido")

    # keep latest preview
    sess["data_html"] = data_html

    return render_template("heatmap_app_00.html",
                           session_id=sid,
                           columns=sess.get("columns"),
                           graph_html=graph_html,
                           form=form_state,
                           chosen_colors=chosen_colors,
                           data_html=data_html)

@app.route("/download/<fmt>/<sid>", methods=["GET"])
def download_plot(fmt, sid):
    """Download the generated plot in the requested format (currently jpg)."""
    fmt = (fmt or "jpg").lower()
    path = os.path.join(PLOT_DIR, f"{sid}.{fmt}")
    if not os.path.exists(path):
        return "File not found. Generate the heatmap first.", 404
    # suggest a friendly filename
    return send_file(path, as_attachment=True, download_name=f"CARBSE_heatmap.{fmt}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
