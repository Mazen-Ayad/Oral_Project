"""Oral Disease Classifier - Gradio app (tuned ResNet50).
 
Put ResNet50_tuned_best.keras next to this file (or set MODEL_PATH), then run:  python app.py
Features: single-image analysis with confidence, Grad-CAM heatmap, PDF report, batch mode with CSV.
"""
import os, csv, tempfile, datetime
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
 
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ResNet50_tuned_best.keras"))
# Same order as meta.json (alphabetical) - the order the model was trained with.
CLASS_NAMES = ["Calculus", "Caries", "Discoloration", "Gingivitis", "Hypodontia", "Mouth_Ulcer"]
TMP = tempfile.mkdtemp(prefix="oral_app_")
 
# Held-out TEST results of this exact model (506 images, notebook step 5).
TEST_METRICS = {
    "Calculus":      {"recall": 0.950, "precision": 0.919, "n": 60},
    "Caries":        {"recall": 0.607, "precision": 0.810, "n": 28},
    "Discoloration": {"recall": 0.739, "precision": 0.773, "n": 23},
    "Gingivitis":    {"recall": 0.973, "precision": 0.957, "n": 183},
    "Hypodontia":    {"recall": 0.994, "precision": 0.978, "n": 180},
    "Mouth_Ulcer":   {"recall": 0.969, "precision": 0.969, "n": 32},
}
OVERALL = {"accuracy": 0.947, "macro_f1": 0.884, "images": 506}
WEAK = {"Caries", "Discoloration"}          # classes with the lowest test recall
 
INFO = {
    "Calculus":      ("Calculus (tartar)", "Hardened plaque on the teeth, usually along the gum line."),
    "Caries":        ("Caries (tooth decay)", "Damage to the tooth surface that can form cavities."),
    "Discoloration": ("Tooth discoloration", "A change in tooth colour, from stains or from inside the tooth."),
    "Gingivitis":    ("Gingivitis", "Red, swollen or bleeding gums caused by inflammation."),
    "Hypodontia":    ("Hypodontia", "One or more teeth missing because they never developed."),
    "Mouth_Ulcer":   ("Mouth ulcer", "A sore on the lining of the mouth, often painful."),
}
COLORS = {"Calculus": "#0ea5e9", "Caries": "#f97316", "Discoloration": "#a855f7",
          "Gingivitis": "#ef4444", "Hypodontia": "#14b8a6", "Mouth_Ulcer": "#eab308"}
CONF_OK, CONF_LOW = 0.75, 0.50               # confidence thresholds for the message shown
 
 
# ----------------------------------------------------------------------------- model
@keras.utils.register_keras_serializable(package="oral")
class ImageNetPreprocess(layers.Layer):
    """Must be registered under the same name as in training, or loading fails."""
    def __init__(self, mode="none", **kwargs):
        super().__init__(**kwargs); self.mode = mode
    def call(self, x):
        if self.mode == "none":
            return x
        return keras.applications.imagenet_utils.preprocess_input(x, mode=self.mode)
    def get_config(self):
        cfg = super().get_config(); cfg.update(mode=self.mode); return cfg
 
 
# Models saved with a newer Keras add a 'quantization_config' key to Dense layers that older Keras
# (3.12 on Python 3.10) rejects. Ignore that key; it does not affect the weights or the predictions.
_orig_dense_init = layers.Dense.__init__
def _dense_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _orig_dense_init(self, *args, **kwargs)
layers.Dense.__init__ = _dense_init
 
model = keras.models.load_model(MODEL_PATH, compile=False)
H, W = model.input_shape[1:3]
assert model.output_shape[-1] == len(CLASS_NAMES), "model output size does not match CLASS_NAMES"
 
# Grad-CAM source: the feature maps of the ResNet50 backbone (a nested model) + the final Dense weights.
# Because the head is GlobalAveragePooling -> Dense, the class activation map is exactly
# sum_k W[k, class] * feature_map_k, so one forward pass gives both the prediction and the heatmap.
# A self-check at start-up compares this path with model.predict; if they differ, the heatmap is switched off.
_pre_model = _base = _W = _b = None
try:
    _pre = next(l for l in model.layers if isinstance(l, ImageNetPreprocess))
    _base = next(l for l in model.layers if hasattr(l, "get_layer") and l is not model)
    _dense = model.layers[-1]
    _W, _b = _dense.get_weights()                                   # (channels, classes), (classes,)
    _pre_model = keras.Model(model.input, _pre.output)
    _test = np.random.default_rng(0).uniform(0, 255, (2, H, W, 3)).astype("float32")
    _f = _base(_pre_model(_test, training=False), training=False).numpy()
    _z = _f.mean((1, 2)) @ _W + _b
    _p = np.exp(_z - _z.max(1, keepdims=True)); _p /= _p.sum(1, keepdims=True)
    if not np.allclose(_p, model.predict(_test, verbose=0), atol=1e-3):
        raise ValueError("Grad-CAM path does not match the model")
except Exception as _e:
    print("Grad-CAM disabled:", _e)
    _pre_model = None
 
 
def predict_one(img):
    """Returns (probabilities[6], cam[h, w] or None)."""
    x = tf.image.resize(tf.convert_to_tensor(np.asarray(img.convert("RGB")), tf.float32), (H, W))[None]
    if _pre_model is None:
        return model.predict(x, verbose=0)[0], None
    feats = _base(_pre_model(x, training=False), training=False).numpy()[0]       # (h, w, channels)
    z = feats.mean((0, 1)) @ _W + _b
    probs = np.exp(z - z.max()); probs /= probs.sum()
    cam = np.tensordot(feats, _W[:, int(np.argmax(probs))], axes=([2], [0]))
    return probs, cam
 
 
def overlay(img, cam):
    """Blend a jet-style heatmap onto the photo; stronger activation = stronger colour."""
    img = img.convert("RGB")
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    m = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(img.size, Image.BICUBIC), np.float32) / 255
    r = np.clip(1.5 - np.abs(4 * m - 3), 0, 1); g = np.clip(1.5 - np.abs(4 * m - 2), 0, 1); b = np.clip(1.5 - np.abs(4 * m - 1), 0, 1)
    heat = np.stack([r, g, b], -1) * 255
    a = (0.65 * m)[..., None]
    out = (1 - a) * np.asarray(img, np.float32) + a * heat
    return Image.fromarray(out.astype("uint8"))
 
 
# ----------------------------------------------------------------------------- UI pieces
def _confidence(conf, second_name, conf2, top_name):
    if conf >= CONF_OK:
        return "ok", f"High confidence ({conf:.0%})"
    if conf >= CONF_LOW:
        return "mid", f"Moderate confidence ({conf:.0%}). Second most likely: <b>{second_name}</b> ({conf2:.0%})."
    return "low", (f"Low confidence ({conf:.0%}). The model is unsure between <b>{top_name}</b> and "
                   f"<b>{second_name}</b>. Try a clearer, closer, well-lit photo.")
 
 
def _card(kind, title, body):
    return f'<div class="res {kind}"><div class="res-title">{title}</div><div class="res-body">{body}</div></div>'
 
 
EMPTY = _card("idle", "No image yet", "Upload a clear, well-lit photo of the teeth or mouth, then press <b>Analyze</b>.")
 
 
def bars_html(probs):
    rows = []
    for i in np.argsort(probs)[::-1]:
        c = CLASS_NAMES[i]; p = float(probs[i])
        rows.append(f'<div class="bar"><div class="bl"><span>{INFO[c][0]}</span><b>{p:.1%}</b></div>'
                    f'<div class="bt"><div class="bf" style="width:{max(p * 100, 0.6):.1f}%;background:{COLORS[c]}"></div></div></div>')
    return '<div class="bars">' + "".join(rows) + "</div>"
 
 
def make_pdf(img, heat, probs, top, kind_msg):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    path = os.path.join(TMP, f"report_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf")
    c = canvas.Canvas(path, pagesize=A4); w, h = A4
    c.setFillColorRGB(0.06, 0.46, 0.43); c.rect(0, h - 70, w, 70, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 20); c.drawString(40, h - 42, "Oral Disease Classifier - Report")
    c.setFont("Helvetica", 10); c.drawString(40, h - 58, f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}")
    y = h - 100
    for im, label in ((img, "Uploaded photo"), (heat, "Model attention (Grad-CAM)")):
        if im is None:
            continue
        s = min(240 / im.width, 240 / im.height)
        iw, ih = im.width * s, im.height * s
        x0 = 40 if label == "Uploaded photo" else 310
        c.drawImage(ImageReader(im.convert("RGB")), x0, y - ih, iw, ih)
        c.setFillColorRGB(0.3, 0.3, 0.3); c.setFont("Helvetica", 9); c.drawString(x0, y - 256, label)
    y -= 290
    c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 16); c.drawString(40, y, f"Most likely finding: {INFO[top][0]}")
    c.setFont("Helvetica", 11); y -= 20; c.drawString(40, y, INFO[top][1])
    y -= 18; c.drawString(40, y, kind_msg)
    y -= 30; c.setFont("Helvetica-Bold", 12); c.drawString(40, y, "Probability of each class"); c.setFont("Helvetica", 11)
    for i in np.argsort(probs)[::-1]:
        y -= 18; c.drawString(48, y, INFO[CLASS_NAMES[i]][0]); c.drawRightString(300, y, f"{probs[i]:.1%}")
        c.setFillColorRGB(0.85, 0.85, 0.85); c.rect(310, y - 1, 200, 9, stroke=0, fill=1)
        c.setFillColorRGB(0.06, 0.46, 0.43); c.rect(310, y - 1, 200 * float(probs[i]), 9, stroke=0, fill=1); c.setFillColorRGB(0, 0, 0)
    y -= 36; c.setFont("Helvetica-Oblique", 9)
    for line in ("Not a medical diagnosis. This tool is a screening aid built for a university project and can be wrong.",
                 f"Model: tuned ResNet50, test accuracy {OVERALL['accuracy']:.1%}, macro-F1 {OVERALL['macro_f1']:.3f} on {OVERALL['images']} images.",
                 "If you have symptoms or concerns, see a dentist or doctor."):
        c.drawString(40, y, line); y -= 13
    c.save()
    return path
 
 
def quality_check(img):
    """Cheap heuristics (not a medical check). Returns a list of short warnings."""
    warns = []
    if min(img.size) < 224:
        warns.append(f"low resolution ({img.width}x{img.height}px; the model works at {W}x{H})")
    g = img.convert("L"); g.thumbnail((512, 512))
    a = np.asarray(g, np.float32)
    mean = a.mean()
    if mean < 60:
        warns.append("very dark")
    elif mean > 205:
        warns.append("over-exposed / very bright")
    if a.std() < 25:
        warns.append("low contrast")
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    if lap.var() < 40:
        warns.append("possibly blurry")
    return warns
 
 
def analyze(image, history):
    history = history or []
    if image is None:
        return EMPTY, "", None, None, history, history
    probs, cam = predict_one(image)
    order = np.argsort(probs)[::-1]
    top, second = CLASS_NAMES[order[0]], CLASS_NAMES[order[1]]
    conf, conf2 = float(probs[order[0]]), float(probs[order[1]])
    kind, msg = _confidence(conf, INFO[second][0], conf2, INFO[top][0])
    name, desc = INFO[top]
    m = TEST_METRICS[top]
    extra = ""
    warns = quality_check(image)
    if warns:
        extra += ('<div class="note"><b>Image quality:</b> ' + ", ".join(warns) +
                  ". Results can be less reliable; try a sharper, well-lit, close-up photo.</div>")
    if top in WEAK:
        extra += (f'<div class="note">On the test set this class was found in only {m["recall"]:.0%} of cases '
                  f'({m["n"]} images), so treat this result with extra caution.</div>')
    card = _card(kind, "Most likely finding",
                 f'<div class="big">{name}</div><div>{desc}</div><div class="conf">{msg}</div>{extra}')
    heat = overlay(image, cam) if cam is not None else None
    plain = msg.replace("<b>", "").replace("</b>", "")
    try:
        pdf = make_pdf(image, heat, probs, top, plain)
    except Exception as e:                      # e.g. reportlab not installed: the analysis still works
        print("PDF report skipped:", repr(e))
        pdf = None
    history = history + [[len(history) + 1, f"{datetime.datetime.now():%H:%M:%S}", name, f"{conf:.1%}",
                          "; ".join(warns) if warns else "ok"]]
    return card, bars_html(probs), heat, pdf, history, history
 
 
def history_csv(history):
    if not history:
        return None
    out = os.path.join(TMP, "history.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["#", "time", "prediction", "confidence", "image quality"]); wr.writerows(history)
    return out
 
 
def analyze_batch(files):
    if not files:
        return [], None
    rows = []
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        try:
            probs, _ = predict_one(Image.open(path))
        except Exception:
            rows.append([os.path.basename(path), "unreadable image", "", ""]); continue
        o = np.argsort(probs)[::-1]
        rows.append([os.path.basename(path), INFO[CLASS_NAMES[o[0]]][0], f"{probs[o[0]]:.1%}",
                     f"{INFO[CLASS_NAMES[o[1]]][0]} ({probs[o[1]]:.1%})"])
    out = os.path.join(TMP, "predictions.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["file", "prediction", "confidence", "second_choice"]); wr.writerows(rows)
    return rows, out
 
 
def metrics_html():
    rows = "".join(
        f'<tr><td>{INFO[c][0]}</td><td>{m["n"]}</td><td>{m["recall"]:.0%}</td><td>{m["precision"]:.0%}</td></tr>'
        for c, m in TEST_METRICS.items())
    return (f'<div class="kpis"><div><b>{OVERALL["accuracy"]:.1%}</b><span>accuracy</span></div>'
            f'<div><b>{OVERALL["macro_f1"]:.3f}</b><span>macro-F1</span></div>'
            f'<div><b>{OVERALL["images"]}</b><span>test images</span></div></div>'
            '<table class="mt"><thead><tr><th>Class</th><th>Test images</th><th>Found (recall)</th>'
            f'<th>Correct when predicted (precision)</th></tr></thead><tbody>{rows}</tbody></table>'
            '<p class="small">Evaluated once on held-out images the model never saw. Caries and Discoloration have the fewest '
            'examples and the lowest recall, so their predictions are the least reliable. Photos from other cameras or clinics '
            'may give lower accuracy.</p>')
 
 
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
.banner {background: linear-gradient(120deg,#0f766e,#0369a1); color:#fff; border-radius:16px; padding:22px 26px; margin-bottom:6px;}
.banner h1 {margin:0; font-size:28px; color:#fff;}
.banner p {margin:6px 0 0; opacity:.9; color:#fff;}
.banner .chips {margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;}
.banner .chips span {background:rgba(255,255,255,.18); border-radius:999px; padding:3px 12px; font-size:13px;}
.res {border-radius:14px; padding:16px 18px; border:1px solid rgba(128,128,128,.3); line-height:1.5;}
.res-title {font-size:12px; text-transform:uppercase; letter-spacing:.06em; opacity:.7; margin-bottom:4px;}
.res .big {font-size:28px; font-weight:700; margin:2px 0 4px;}
.res .conf {margin-top:10px; font-weight:600;}
.res .note {margin-top:10px; padding:8px 10px; border-radius:8px; background:rgba(245,158,11,.18); font-size:14px;}
.res.ok {border-left:6px solid #16a34a;} .res.mid {border-left:6px solid #f59e0b;}
.res.low {border-left:6px solid #dc2626;} .res.idle {border-left:6px solid #9ca3af;}
.bars {padding:6px 2px;}
.bar {margin:9px 0;} .bl {display:flex; justify-content:space-between; font-size:14px; margin-bottom:3px;}
.bt {height:10px; border-radius:6px; background:rgba(128,128,128,.22); overflow:hidden;}
.bf {height:100%; border-radius:6px;}
.kpis {display:flex; gap:14px; margin:6px 0 14px; flex-wrap:wrap;}
.kpis div {flex:1; min-width:140px; border:1px solid rgba(128,128,128,.3); border-radius:12px; padding:12px 16px;}
.kpis b {display:block; font-size:26px;} .kpis span {opacity:.7; font-size:13px;}
.mt {width:100%; border-collapse:collapse;}
.mt th,.mt td {padding:8px 10px; border-bottom:1px solid rgba(128,128,128,.3); text-align:left;}
.small {font-size:13px; opacity:.8;}
.disc {font-size:13px; opacity:.85; padding-top:6px;}
"""
 
with gr.Blocks(title="Oral Disease Classifier") as demo:
    gr.HTML('<div class="banner"><h1>Oral Disease Classifier</h1>'
            '<p>Image classification of six oral conditions with a fine-tuned ResNet50.</p>'
            f'<div class="chips"><span>Test accuracy {OVERALL["accuracy"]:.1%}</span><span>Macro-F1 {OVERALL["macro_f1"]:.3f}</span>'
            '<span>Grad-CAM heatmap</span><span>PDF report</span></div></div>')
    hist = gr.State([])
    hist_view = gr.Dataframe(headers=["#", "time", "prediction", "confidence", "image quality"],
                             interactive=False, wrap=True, render=False)
    with gr.Tabs():
        with gr.Tab("Analyze"):
            with gr.Row():
                with gr.Column(scale=5):
                    img = gr.Image(type="pil", label="Photo of the teeth / mouth", height=400)
                    with gr.Row():
                        clear = gr.Button("Clear")
                        go = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=6):
                    result = gr.HTML(EMPTY)
                    with gr.Tabs():
                        with gr.Tab("Probabilities"):
                            bars = gr.HTML("")
                        with gr.Tab("Where the model looked"):
                            heat = gr.Image(label="Grad-CAM (warm colours = areas that most influenced the prediction)",
                                            interactive=False, height=320)
                            gr.HTML('<p class="small">The heatmap shows what drove the model, not the exact location of a lesion. '
                                    'If it highlights the background, distrust the result.</p>')
                    pdf = gr.File(label="Download report (PDF)", interactive=False)
            outs = [result, bars, heat, pdf, hist, hist_view]
            go.click(analyze, [img, hist], outs)
            img.upload(analyze, [img, hist], outs)
            clear.click(lambda: (None, EMPTY, "", None, None), None, [img, result, bars, heat, pdf])
        with gr.Tab("Batch"):
            gr.HTML('<p class="small">Upload several photos to classify them all at once, then download the results as a CSV file.</p>')
            files = gr.File(file_count="multiple", file_types=["image"], label="Photos")
            run = gr.Button("Classify all", variant="primary")
            table = gr.Dataframe(headers=["file", "prediction", "confidence", "second choice"], interactive=False, wrap=True)
            csv_out = gr.File(label="Download results (CSV)", interactive=False)
            run.click(analyze_batch, files, [table, csv_out])
        with gr.Tab("History"):
            gr.HTML('<p class="small">Every photo analysed in this session. The list is cleared when you close the page.</p>')
            hist_view.render()
            with gr.Row():
                hist_csv_btn = gr.Button("Export CSV")
                hist_clear = gr.Button("Clear history")
            hist_file = gr.File(label="Download history (CSV)", interactive=False)
        with gr.Tab("Model performance"):
            gr.HTML(metrics_html())
    hist_csv_btn.click(history_csv, hist, hist_file)
    hist_clear.click(lambda: ([], [], None), None, [hist, hist_view, hist_file])
    gr.HTML('<div class="disc"><b>Not a medical diagnosis.</b> This tool is a screening aid built for a '
            'university project. It can be wrong. If you have symptoms or concerns, see a dentist or doctor.</div>')
 
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"), css=CSS)
 
