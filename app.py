"""
BC5CDR NER + Relation Extraction Pipeline — Streamlit App
PubMedBERT · Chemical/Disease NER · CID Relation Extraction
"""

import os
import re
import math
import json
import torch
import numpy as np
import streamlit as st
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BC5CDR · NER + RE Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0b0e17;
    --surface:   #111520;
    --border:    #1e2535;
    --chem-bg:   rgba(0, 200, 150, 0.12);
    --chem-text: #00c896;
    --chem-bdr:  #00c896;
    --dis-bg:    rgba(255, 107, 107, 0.12);
    --dis-text:  #ff6b6b;
    --dis-bdr:   #ff6b6b;
    --accent:    #4fc3f7;
    --muted:     #4a5568;
    --text:      #e2e8f0;
    --subtext:   #718096;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* header strip */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.app-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.45rem; font-weight: 700;
    color: var(--accent); margin: 0; letter-spacing: -0.5px;
}
.app-header p { color: var(--subtext); margin: 0; font-size: 0.82rem; }

/* stat cards */
.stat-row { display: flex; gap: 12px; margin-bottom: 24px; }
.stat-card {
    flex: 1; background: var(--surface);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 20px; text-align: center;
}
.stat-card .num {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700; line-height: 1;
}
.stat-card .lbl {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--subtext); margin-top: 4px;
}
.stat-card.chem .num { color: var(--chem-text); }
.stat-card.dis  .num { color: var(--dis-text);  }
.stat-card.rel  .num { color: var(--accent);    }

/* annotated text */
.ann-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px;
    font-size: 0.95rem; line-height: 2;
    margin-bottom: 20px;
}
.tag-chem {
    background: var(--chem-bg); color: var(--chem-text);
    border: 1px solid var(--chem-bdr);
    border-radius: 4px; padding: 1px 7px; margin: 0 2px;
    font-weight: 600; font-size: 0.88rem;
}
.tag-chem sup { font-size: 0.6rem; color: var(--chem-text); opacity: 0.8; }
.tag-dis {
    background: var(--dis-bg); color: var(--dis-text);
    border: 1px solid var(--dis-bdr);
    border-radius: 4px; padding: 1px 7px; margin: 0 2px;
    font-weight: 600; font-size: 0.88rem;
}
.tag-dis sup { font-size: 0.6rem; color: var(--dis-text); opacity: 0.8; }

/* relation cards */
.rel-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px; padding: 12px 18px;
    margin-bottom: 10px; display: flex;
    align-items: center; justify-content: space-between;
}
.rel-card .arrow { color: var(--accent); font-family: 'Space Mono', monospace; font-size: 0.85rem; }
.rel-card .conf { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: var(--subtext); }

/* section labels */
.section-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; letter-spacing: 2px;
    text-transform: uppercase; color: var(--subtext);
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* buttons */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00c896 0%, #4fc3f7 100%);
    color: #0b0e17; font-weight: 700; border: none;
    border-radius: 8px; padding: 0.55rem 1.6rem;
    font-family: 'Space Mono', monospace; font-size: 0.82rem;
    letter-spacing: 0.5px; transition: opacity 0.2s;
    width: 100%;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85; }

/* text area */
div[data-testid="stTextArea"] textarea {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,195,247,0.15) !important;
}

/* sliders */
div[data-testid="stSlider"] { padding: 4px 0; }

/* selectbox */
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* expander */
details { background: var(--surface) !important; border-color: var(--border) !important; }
summary { color: var(--text) !important; }

/* status badge */
.badge-ready {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    background: rgba(0,200,150,0.12); border: 1px solid #00c896;
    color: #00c896; font-size: 0.75rem; font-family: 'Space Mono', monospace;
}
.badge-warn {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    background: rgba(255,152,0,0.12); border: 1px solid #ff9800;
    color: #ff9800; font-size: 0.75rem; font-family: 'Space Mono', monospace;
}

div[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODEL_NAME   = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
LABEL2ID     = {'O':0,'B-Chemical':1,'I-Chemical':2,'B-Disease':3,'I-Disease':4}
ID2LABEL     = {v: k for k, v in LABEL2ID.items()}
SPECIAL_TOKS = ['[CHEM]','[/CHEM]','[DIS]','[/DIS]']
MAX_LEN      = 512

EXAMPLES = [
    {
        "label": "Example 1 — Cisplatin nephrotoxicity",
        "text": "Cisplatin-induced nephrotoxicity is a major dose-limiting side effect, while carboplatin causes less renal damage but more thrombocytopenia."
    },
    {
        "label": "Example 2 — Methotrexate hepatotoxicity",
        "text": "Methotrexate treatment was associated with hepatotoxicity and bone marrow suppression, whereas vincristine caused peripheral neuropathy."
    },
    {
        "label": "Example 3 — Amiodarone toxicity",
        "text": "Long-term amiodarone therapy can lead to pulmonary toxicity and thyroid dysfunction; digoxin overdose is associated with cardiac arrhythmia."
    },
]

# ─────────────────────────────────────────────
# RE MODEL DEFINITION  (must match training)
# ─────────────────────────────────────────────
class REEntityModel(nn.Module):
    def __init__(self, model_name, num_labels, tokenizer):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        hidden = self.bert.config.hidden_size
        self.dropout    = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden * 2, num_labels)
        self.chem_id    = tokenizer.convert_tokens_to_ids('[CHEM]')
        self.dis_id     = tokenizer.convert_tokens_to_ids('[DIS]')

    def forward(self, input_ids, attention_mask, labels=None):
        out        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_st  = out.last_hidden_state
        chem_vecs, dis_vecs = [], []
        for i in range(input_ids.size(0)):
            ids      = input_ids[i]
            cp       = (ids == self.chem_id).nonzero(as_tuple=True)[0]
            dp       = (ids == self.dis_id ).nonzero(as_tuple=True)[0]
            chem_vecs.append(hidden_st[i, cp[0] if len(cp) > 0 else 0])
            dis_vecs .append(hidden_st[i, dp[0] if len(dp) > 0 else 0])
        chem_vecs = torch.stack(chem_vecs)
        dis_vecs  = torch.stack(dis_vecs)
        logits    = self.classifier(self.dropout(torch.cat([chem_vecs, dis_vecs], dim=1)))
        loss      = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False, ttl=0)
def load_models(ner_path, re_path):
    errors = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- NER ----------------
    # ========================= FIX NER META TENSOR LOAD ERROR =========================
# Replace ONLY your NER block inside load_models() with this

    # ---------------- NER ----------------
    try:
        ner_tok = AutoTokenizer.from_pretrained(ner_path)

        # Load with low_cpu_mem_usage=False to avoid meta tensor issue
        ner_model = AutoModelForTokenClassification.from_pretrained(
            ner_path,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False
        )

        # Force proper label maps
        ner_model.config.id2label = ID2LABEL
        ner_model.config.label2id = LABEL2ID

        # Move safely
        ner_model = ner_model.to(device)

        ner_model.eval()

    except Exception as ex:
        ner_tok = None
        ner_model = None
        errors.append(f"NER load error: {str(ex)}")

    # ========================= FIX RE TOKENIZER PATH NoneType ERROR =========================
# Root cause:
# tokenizer files inside ./re_model are broken/missing, so AutoTokenizer fails.
# Use base tokenizer fallback if local tokenizer config is corrupted.

# Replace ONLY your RE block inside load_models() with this:


    # ---------------- RE ----------------
    try:
        # -------- TRY LOCAL TOKENIZER FIRST --------
        try:
            re_tok = AutoTokenizer.from_pretrained(
                re_path,
                use_fast=False
            )

        # -------- FALLBACK TO BASE MODEL --------
        except Exception:
            re_tok = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                use_fast=False
            )

        # Ensure special tokens exist
        missing_tokens = [
            tok for tok in SPECIAL_TOKS
            if tok not in re_tok.get_vocab()
        ]

        if missing_tokens:
            re_tok.add_special_tokens({
                "additional_special_tokens": missing_tokens
            })

        # -------- BUILD MODEL --------
        re_mdl = REEntityModel(
            MODEL_NAME,
            num_labels=2,
            tokenizer=re_tok
        )

        weights_file = os.path.join(
            re_path,
            "re_model.pt"
        )

        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"Missing RE weights: {weights_file}"
            )

        # -------- LOAD CHECKPOINT --------
        state_dict = torch.load(
            weights_file,
            map_location=device
        )

        re_mdl.load_state_dict(state_dict)

        re_mdl = re_mdl.to(device)

        re_mdl.eval()

    except Exception as ex:
        re_tok = None
        re_mdl = None
        errors.append(f"RE load error: {str(ex)}")

    return (
        ner_tok,
        ner_model,
        re_tok,
        re_mdl,
        device,
        errors
    )

# ─────────────────────────────────────────────
# NER INFERENCE
# ─────────────────────────────────────────────
def run_ner(text, ner_tok, ner_model, conf_thresh=0.55):
    enc = ner_tok(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_LEN, return_offsets_mapping=True
    )
    offsets = enc.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        out    = ner_model(**enc)
        probs  = torch.softmax(out.logits[0], dim=-1)
        labels = torch.argmax(probs, dim=-1).tolist()
        confs  = probs.max(dim=-1).values.tolist()

    entities, current = [], None
    for idx, (label_id, conf, (s, e)) in enumerate(zip(labels, confs, offsets)):
        if s == e:
            continue
        label = ID2LABEL.get(label_id, "O")
        if label.startswith("B-") and conf >= conf_thresh:
            if current:
                entities.append(current)
            current = {"type": label[2:], "start": s, "end": e,
                       "text": text[s:e], "conf": conf}
        elif label.startswith("I-") and current and conf >= conf_thresh:
            current["end"]  = e
            current["text"] = text[current["start"]:e]
            current["conf"] = min(current["conf"], conf)
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return entities

# ─────────────────────────────────────────────
# INSERT ENTITY MARKERS (fixed offset tracking)
# ─────────────────────────────────────────────
def insert_markers(text, chem, disease):
    spans = sorted([
        (chem["start"],    chem["end"],    "[CHEM]", "[/CHEM]"),
        (disease["start"], disease["end"], "[DIS]",  "[/DIS]"),
    ], key=lambda x: x[0])
    result, offset = text, 0
    for (s, e, open_t, close_t) in spans:
        s2, e2       = s + offset, e + offset
        entity_text  = result[s2:e2]
        inserted     = f"{open_t}{entity_text}{close_t}"
        result       = result[:s2] + inserted + result[e2:]
        offset      += len(open_t) + len(close_t)   # no +2 (correct fix)
    return result

# ─────────────────────────────────────────────
# RE INFERENCE
# ─────────────────────────────────────────────
# ========================= REPLACE run_re() WITH THIS =========================

def run_re(text, chems, diseases, re_tok, re_mdl, conf_thresh=0.50):
    relations = []

    device = next(re_mdl.parameters()).device

    for c in chems:
        for d in diseases:

            marked = insert_markers(text, c, d)

            enc = re_tok(
                marked,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt"
            )

            # ---------------- FIX ----------------
            # Remove unsupported token_type_ids
            if "token_type_ids" in enc:
                enc.pop("token_type_ids")

            # Move to model device
            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            with torch.no_grad():
                out = re_mdl(**enc)

                logits = (
                    out["logits"]
                    if isinstance(out, dict)
                    else out.logits
                )

                probs = torch.softmax(
                    logits,
                    dim=-1
                )[0]

            pred = torch.argmax(probs).item()

            # Positive CID class probability
            conf = probs[1].item()

            if pred == 1 and conf >= conf_thresh:

                relations.append({
                    "chemical": c["text"],
                    "disease": d["text"],
                    "conf": round(conf, 4),
                    "chem_ent": c,
                    "dis_ent": d,
                })

    relations.sort(
        key=lambda x: -x["conf"]
    )

    return relations

# ─────────────────────────────────────────────
# ANNOTATED TEXT HTML
# ─────────────────────────────────────────────
def build_annotated_html(text, entities):
    sorted_ents = sorted(entities, key=lambda x: x["start"])
    html, cursor = "", 0
    for ent in sorted_ents:
        s, e   = ent["start"], ent["end"]
        plain  = text[cursor:s].replace("<", "&lt;").replace(">", "&gt;")
        ename  = ent["text"].replace("<", "&lt;").replace(">", "&gt;")
        label  = ent["type"][:3].upper()
        tag    = "tag-chem" if ent["type"] == "Chemical" else "tag-dis"
        html  += plain + f'<span class="{tag}">{ename}<sup>{label}</sup></span>'
        cursor = e
    html += text[cursor:].replace("<", "&lt;").replace(">", "&gt;")
    return html

# ─────────────────────────────────────────────
# KNOWLEDGE GRAPH
# ─────────────────────────────────────────────
def draw_knowledge_graph(entities, relations):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0b0e17")
    ax.set_facecolor("#0b0e17")
    ax.axis("off")

    if not entities and not relations:
        ax.text(0.5, 0.5, "No entities detected", ha="center", va="center",
                color="#4a5568", fontsize=12, transform=ax.transAxes)
        return fig

    G = nx.DiGraph()

    chems = [e["text"] for e in entities if e["type"] == "Chemical"]
    discs = [e["text"] for e in entities if e["type"] == "Disease"]

    for c in set(chems): G.add_node(c, kind="chem")
    for d in set(discs): G.add_node(d, kind="dis")
    for r in relations:
        G.add_edge(r["chemical"], r["disease"], conf=r["conf"])

    if len(G.nodes) == 0:
        return fig

    # layout — bipartite-style
    pos = {}
    chem_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "chem"]
    dis_nodes  = [n for n, d in G.nodes(data=True) if d.get("kind") == "dis"]

    for i, n in enumerate(chem_nodes):
        pos[n] = (0.15, 1 - (i + 1) / (len(chem_nodes) + 1))
    for i, n in enumerate(dis_nodes):
        pos[n] = (0.85, 1 - (i + 1) / (len(dis_nodes) + 1))

    # draw edges
    for u, v, data in G.edges(data=True):
        conf  = data.get("conf", 0.5)
        alpha = 0.3 + 0.65 * conf
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=["#4fc3f7"], alpha=alpha,
            arrows=True, arrowsize=18,
            connectionstyle="arc3,rad=0.1",
            width=1.5 + conf,
            node_size=1800,
        )
        # conf label
        mx = (pos[u][0] + pos[v][0]) / 2
        my = (pos[u][1] + pos[v][1]) / 2
        ax.text(mx, my, f"{conf:.2f}", fontsize=7,
                color="#4fc3f7", ha="center", va="center",
                bbox=dict(fc="#111520", ec="none", pad=1.5))

    # draw nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=chem_nodes, ax=ax,
        node_color="#00c896", node_size=1800, alpha=0.85
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=dis_nodes, ax=ax,
        node_color="#ff6b6b", node_size=1800, alpha=0.85
    )

    # labels (wrap long names)
    def wrap(s, n=14):
        return "\n".join([s[i:i+n] for i in range(0, len(s), n)])

    labels = {n: wrap(n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8, font_color="white", font_weight="bold"
    )

    # legend
    leg = [
        mpatches.Patch(color="#00c896", label="Chemical"),
        mpatches.Patch(color="#ff6b6b", label="Disease"),
        mpatches.Patch(color="#4fc3f7", label="CID Relation"),
    ]
    ax.legend(handles=leg, loc="lower center", ncol=3,
              facecolor="#111520", edgecolor="#1e2535",
              labelcolor="white", fontsize=8,
              bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# CONFIDENCE BAR CHART
# ─────────────────────────────────────────────
def draw_confidence_chart(relations):
    if not relations:
        return None
    fig, ax = plt.subplots(figsize=(7, max(2, len(relations) * 0.6 + 1)))
    fig.patch.set_facecolor("#0b0e17")
    ax.set_facecolor("#111520")

    labels = [f"{r['chemical']} → {r['disease']}" for r in relations]
    confs  = [r["conf"] for r in relations]
    colors = ["#00c896" if c >= 0.7 else "#4fc3f7" if c >= 0.5 else "#ff9800"
              for c in confs]

    bars = ax.barh(labels, confs, color=colors, edgecolor="none", height=0.55)
    for bar, c in zip(bars, confs):
        ax.text(min(c + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                f"{c:.3f}", va="center", color="white",
                fontsize=8, fontfamily="monospace")

    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color="#ff9800", linestyle="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Confidence", color="#718096", fontsize=8)
    ax.tick_params(colors="#718096", labelsize=8)
    ax.spines[:].set_color("#1e2535")
    ax.set_title("CID Relation Confidences", color="#e2e8f0",
                 fontsize=9, fontfamily="monospace")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# ENTITY DISTRIBUTION CHART
# ─────────────────────────────────────────────
def draw_entity_chart(entities):
    chems = [e for e in entities if e["type"] == "Chemical"]
    discs = [e for e in entities if e["type"] == "Disease"]
    if not chems and not discs:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    fig.patch.set_facecolor("#0b0e17")

    for ax, ents, color, title in [
        (axes[0], chems, "#00c896", "Chemicals"),
        (axes[1], discs, "#ff6b6b", "Diseases"),
    ]:
        ax.set_facecolor("#111520")
        if ents:
            ax.barh([e["text"][:20] for e in ents],
                    [e["conf"] for e in ents],
                    color=color, alpha=0.8, edgecolor="none", height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.55, color="white", linestyle="--", lw=0.7, alpha=0.4)
        ax.set_title(title, color="#e2e8f0", fontsize=8, fontfamily="monospace")
        ax.tick_params(colors="#718096", labelsize=7)
        ax.spines[:].set_color("#1e2535")
        ax.set_xlabel("NER Confidence", color="#718096", fontsize=7)

    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────


with st.sidebar:
    st.markdown("### ⚙️ Model Paths")
    ner_path = st.text_input("NER model folder", value="./ner_model")
    re_path  = st.text_input("RE model folder",  value="./re_model")
    st.write("NER path exists:", os.path.exists(ner_path))
    st.write("RE path exists:", os.path.exists(re_path))
    st.write("RE weights exists:", os.path.exists(os.path.join(re_path, "re_model.pt")))

    st.markdown("---")
    st.markdown("### 🎚️ Thresholds")
    ner_thresh = st.slider("NER confidence", 0.1, 0.95, 0.55, 0.01,
                           help="Minimum token-level confidence to accept an entity")
    re_thresh  = st.slider("RE confidence",  0.1, 0.95, 0.50, 0.01,
                           help="Minimum confidence to report a CID relation")

    st.markdown("---")
    load_btn = st.button("⚡ Load Models")

    # ── model state ──────────────────────────
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False

    if load_btn:
        st.cache_resource.clear()

        with st.spinner("Loading PubMedBERT weights…"):
            nt, nm, rt, rm, dev, errs = load_models(
                ner_path,
                re_path
            )

            st.session_state.ner_tok = nt
            st.session_state.ner_model = nm
            st.session_state.re_tok = rt
            st.session_state.re_model = rm
            st.session_state.device = dev
            st.session_state.load_errs = errs

            st.session_state.models_loaded = (
                nt is not None and
                nm is not None and
                rt is not None and
                rm is not None
            )

    if st.session_state.models_loaded:
        st.success("✅ Models loaded successfully")
    else:
        st.error("❌ Model loading failed")
        

    if st.session_state.models_loaded:
        st.markdown('<span class="badge-ready">✓ Models ready</span>',
                    unsafe_allow_html=True)
        if st.session_state.get("load_errs"):
            for e in st.session_state.load_errs:
                st.warning(e)
    else:
        st.markdown('<span class="badge-warn">⚠ Not loaded</span>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏷️ NER label map")
    for k, v in LABEL2ID.items():
        color = "#00c896" if "Chem" in k else "#ff6b6b" if "Dis" in k else "#4a5568"
        st.markdown(
            f'<span style="font-family:monospace;font-size:0.78rem;color:{color}">'
            f'{v}: {k}</span>', unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span style="font-size:2rem">🧬</span>
  <div>
    <h1>BC5CDR · NER + RE Pipeline</h1>
    <p>PubMedBERT · Chemical/Disease Recognition · CID Relation Extraction</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── example selector + text area ─────────────
col_ex, col_run = st.columns([3, 1])
with col_ex:
    example_label = st.selectbox(
        "Load example",
        ["— custom input —"] + [e["label"] for e in EXAMPLES],
        label_visibility="collapsed"
    )

default_text = ""
for ex in EXAMPLES:
    if ex["label"] == example_label:
        default_text = ex["text"]
        break

input_text = st.text_area(
    "Input text",
    value=default_text,
    height=110,
    placeholder="Paste a biomedical sentence or abstract here…",
    label_visibility="collapsed"
)

run_btn = st.button("▶ Run Pipeline", disabled=not st.session_state.get("models_loaded", False))

if not st.session_state.get("models_loaded", False):
    st.info("Load models from the sidebar to run the pipeline.")

# ─────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────
if run_btn and input_text.strip():
    text = input_text.strip()

    with st.spinner("Running NER…"):
        entities = run_ner(
            text,
            st.session_state.ner_tok,
            st.session_state.ner_model,
            conf_thresh=ner_thresh
        )

    chems = [e for e in entities if e["type"] == "Chemical"]
    discs = [e for e in entities if e["type"] == "Disease"]

    with st.spinner("Running RE…"):
        relations = run_re(
            text, chems, discs,
            st.session_state.re_tok,
            st.session_state.re_model,
            conf_thresh=re_thresh
        ) if chems and discs else []

    # ── stat cards ───────────────────────────
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card chem">
        <div class="num">{len(chems)}</div>
        <div class="lbl">Chemicals</div>
      </div>
      <div class="stat-card dis">
        <div class="num">{len(discs)}</div>
        <div class="lbl">Diseases</div>
      </div>
      <div class="stat-card rel">
        <div class="num">{len(relations)}</div>
        <div class="lbl">CID Relations</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── tabs ─────────────────────────────────
    tab_ann, tab_graph, tab_conf, tab_ents, tab_debug = st.tabs([
        "ANNOTATED TEXT", "KNOWLEDGE GRAPH", "CONFIDENCE CHART",
        "ENTITY DETAIL", "DEBUG"
    ])

    # ─── Annotated Text ──────────────────────
    with tab_ann:
        st.markdown('<div class="section-lbl">Annotated output</div>',
                    unsafe_allow_html=True)
        ann_html = build_annotated_html(text, entities)
        st.markdown(f'<div class="ann-box">{ann_html}</div>',
                    unsafe_allow_html=True)

        if relations:
            st.markdown('<div class="section-lbl" style="margin-top:20px">CID Relations detected</div>',
                        unsafe_allow_html=True)
            for r in relations:
                bar_width = int(r["conf"] * 100)
                st.markdown(f"""
                <div class="rel-card">
                  <div>
                    <span style="color:#00c896;font-weight:600">{r['chemical']}</span>
                    <span class="arrow"> ──CID──▶ </span>
                    <span style="color:#ff6b6b;font-weight:600">{r['disease']}</span>
                  </div>
                  <div style="text-align:right">
                    <div class="conf">conf: {r['conf']:.3f}</div>
                    <div style="background:#1e2535;border-radius:4px;height:4px;width:80px;margin-top:4px">
                      <div style="background:#4fc3f7;width:{bar_width}%;height:4px;border-radius:4px"></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:#4a5568;font-size:0.85rem;margin-top:12px">'
                'No CID relations detected above the threshold.</p>',
                unsafe_allow_html=True
            )

    # ─── Knowledge Graph ─────────────────────
    with tab_graph:
        st.markdown('<div class="section-lbl">Entity–relation knowledge graph</div>',
                    unsafe_allow_html=True)
        fig_g = draw_knowledge_graph(entities, relations)
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

        st.caption(
            "Green nodes = Chemicals · Red nodes = Diseases · "
            "Blue edges = CID relations (edge weight = confidence)"
        )

    # ─── Confidence Chart ────────────────────
    with tab_conf:
        st.markdown('<div class="section-lbl">Relation confidence scores</div>',
                    unsafe_allow_html=True)
        fig_c = draw_confidence_chart(relations)
        if fig_c:
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
            st.caption("Orange dashed line = 0.50 threshold. Green ≥ 0.70 · Blue ≥ 0.50 · Orange < 0.50")
        else:
            st.info("No relations to display.")

        st.markdown('<div class="section-lbl" style="margin-top:20px">NER confidence per entity</div>',
                    unsafe_allow_html=True)
        fig_e2 = draw_entity_chart(entities)
        if fig_e2:
            st.pyplot(fig_e2, use_container_width=True)
            plt.close(fig_e2)

    # ─── Entity Detail ───────────────────────
    with tab_ents:
        st.markdown('<div class="section-lbl">Detected entities</div>',
                    unsafe_allow_html=True)
        if entities:
            for ent in entities:
                tag = "tag-chem" if ent["type"] == "Chemical" else "tag-dis"
                conf_pct = int(ent["conf"] * 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:14px;
                            padding:10px 14px;margin-bottom:8px;
                            background:#111520;border:1px solid #1e2535;border-radius:8px">
                  <span class="{tag}">{ent['text']}</span>
                  <span style="color:#718096;font-size:0.78rem;font-family:monospace">
                    {ent['type']} · span [{ent['start']}:{ent['end']}] · conf {ent['conf']:.3f}
                  </span>
                  <div style="margin-left:auto;background:#1e2535;border-radius:4px;
                              height:6px;width:100px">
                    <div style="background:{'#00c896' if ent['type']=='Chemical' else '#ff6b6b'};
                                width:{conf_pct}%;height:6px;border-radius:4px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No entities detected above NER threshold.")

    # ─── Debug ───────────────────────────────
    with tab_debug:
        st.markdown('<div class="section-lbl">Raw token predictions (first 30 tokens)</div>',
                    unsafe_allow_html=True)
        try:
            enc_dbg  = st.session_state.ner_tok(
                text, return_tensors="pt", truncation=True,
                max_length=MAX_LEN, return_offsets_mapping=True
            )
            offsets  = enc_dbg.pop("offset_mapping")[0].tolist()
            with torch.no_grad():
                out_dbg  = st.session_state.ner_model(**enc_dbg)
                probs_dbg = torch.softmax(out_dbg.logits[0], dim=-1)
                preds_dbg = torch.argmax(probs_dbg, dim=-1).tolist()
                confs_dbg = probs_dbg.max(dim=-1).values.tolist()

            tokens_dbg = st.session_state.ner_tok.convert_ids_to_tokens(
                enc_dbg["input_ids"][0].tolist()
            )

            rows = []
            for tok, pred, conf, (s, e) in zip(
                tokens_dbg[:30], preds_dbg[:30], confs_dbg[:30], offsets[:30]
            ):
                rows.append({
                    "Token": tok,
                    "Label": ID2LABEL.get(pred, "O"),
                    "Conf":  f"{conf:.3f}",
                    "Span":  f"[{s}:{e}]",
                })

            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as ex:
            st.error(f"Debug render error: {ex}")

        st.markdown('<div class="section-lbl" style="margin-top:20px">Marked RE inputs (first 3 pairs)</div>',
                    unsafe_allow_html=True)
        count = 0
        for c in chems:
            for d in discs:
                if count >= 3:
                    break
                marked = insert_markers(text, c, d)
                st.code(marked, language=None)
                count += 1

elif run_btn and not input_text.strip():
    st.warning("Please enter some text before running the pipeline.")