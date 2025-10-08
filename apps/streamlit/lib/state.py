import pathlib
from typing import Optional
from PIL import Image
import streamlit as st

def init_state():
    ss = st.session_state
    ss.setdefault("mode_top", "Single Model (MATFHI)")
    ss.setdefault("submode", "Predict Only")
    ss.setdefault("selected_stem", None)
    ss.setdefault("sel_idx", 0)
    ss.setdefault("files_img", [])
    ss.setdefault("fov_by_stem", {})
    ss.setdefault("files_gt", [])
    ss.setdefault("files_fov", [])
    ss.setdefault("results", {})
    ss.setdefault("messages", [])
    ss.setdefault("running", False)
    ss.setdefault("stop_flag", False)
    ss.setdefault("done_once", False)
    ss.setdefault("deleted_stems", set())
    ss.setdefault("uploader_nonce", 0)
    ss.setdefault("fov_uploader_nonce", {})
    ss.setdefault("overlay_tog", False)
    ss.setdefault("fov_tog", False)
    ss.setdefault("overlay_reset", False)
    ss.setdefault("gt_by_stem", {})
    ss.setdefault("dataset_choice", "DRIVE")
    ss.setdefault("gt_uploader_nonce", {})   # per-stem nonce to reset GT uploader

def add_msg(kind: str, text: str):
    st.session_state["messages"].append({"kind": kind, "text": text})

def pil_from_upload(f) -> Image.Image:
    return Image.open(f).convert("RGB")

def stem_of(name: str) -> str:
    return pathlib.Path(name).stem

def clear_session_outputs():
    st.session_state["results"] = {}
    st.session_state["messages"] = []
    st.session_state["selected_stem"] = None
    st.session_state["running"] = False
    st.session_state["stop_flag"] = False
    st.session_state["done_once"] = False
    st.session_state["sel_idx"] = 0
    st.session_state["fov_by_stem"] = {}
    st.session_state["deleted_stems"] = set()
    st.session_state["uploader_nonce"] += 1
    st.session_state["fov_uploader_nonce"] = {}

def delete_image_by_stem(stem: str):
    from .state import stem_of  # avoid circulars if you move things later
    st.session_state["files_img"] = [
        f for f in st.session_state.get("files_img", []) if stem_of(f.name) != stem
    ]
    st.session_state["deleted_stems"].add(stem)
    st.session_state["fov_by_stem"].pop(stem, None)
    st.session_state["results"].pop(stem, None)
    if st.session_state.get("selected_stem") == stem:
        st.session_state["selected_stem"] = None
    n = len(st.session_state.get("files_img", []))
    st.session_state["sel_idx"] = 0 if n == 0 else min(st.session_state["sel_idx"], n - 1)
    st.session_state["uploader_nonce"] += 1
    st.session_state["fov_uploader_nonce"].pop(stem, None)
    st.session_state["gt_by_stem"].pop(stem, None)
