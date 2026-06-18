"""Patch pybioclip to support BioCLIP 2.5 Huge (ViT-H/14).

pybioclip 2.1.5 only knows about bioclip and bioclip-2 in its TOL_MODELS
dict (in _constants.py), and hardcodes txt_emb_species.{npy,json} filenames
in predict.py.  BioCLIP 2.5 text embeddings live in the same TreeOfLife-200M
dataset repo but under model-specific filenames.

This script patches both files in-place.
"""
import pathlib
import bioclip._constants
import bioclip.predict

# --- Patch _constants.py: add BioCLIP 2.5 to TOL_MODELS ---
const_path = pathlib.Path(bioclip._constants.__file__)
const_src = const_path.read_text()

old_tol = """TOL_MODELS = {
    BIOCLIP_V1_MODEL_STR: TOL10M_HF_DATAFILE_REPO,
    BIOCLIP_V2_MODEL_STR: TOL200M_HF_DATAFILE_REPO
}"""

new_tol = """BIOCLIP_V25_MODEL_STR = "hf-hub:imageomics/bioclip-2.5-vith14"
TOL_MODELS = {
    BIOCLIP_V1_MODEL_STR: TOL10M_HF_DATAFILE_REPO,
    BIOCLIP_V2_MODEL_STR: TOL200M_HF_DATAFILE_REPO,
    BIOCLIP_V25_MODEL_STR: TOL200M_HF_DATAFILE_REPO,
}

# Map model strings to their specific embedding filenames.
# Models not listed here fall back to the default txt_emb_species files.
TOL_EMB_FILES = {
    BIOCLIP_V25_MODEL_STR: ("txt_emb_bioclip-2.5-vith14.npy", "txt_emb_bioclip-2.5-vith14.json"),
    BIOCLIP_V2_MODEL_STR: ("txt_emb_bioclip-2.npy", "txt_emb_bioclip-2.json"),
}"""

assert old_tol in const_src, f"Could not find TOL_MODELS in _constants.py"
const_src = const_src.replace(old_tol, new_tol)
const_path.write_text(const_src)
print("Patched _constants.py: added BioCLIP 2.5 to TOL_MODELS")

# --- Patch predict.py: use model-specific embedding filenames ---
pred_path = pathlib.Path(bioclip.predict.__file__)
pred_src = pred_path.read_text()

# Add TOL_EMB_FILES to the imports from _constants
old_import = "    HF_DATAFILE_REPO_TYPE, BIOCLIP_MODEL_STR, TOL_MODELS,"
new_import = "    HF_DATAFILE_REPO_TYPE, BIOCLIP_MODEL_STR, TOL_MODELS, TOL_EMB_FILES,"
assert old_import in pred_src, "Could not find TOL_MODELS import in predict.py"
pred_src = pred_src.replace(old_import, new_import)

# Override get_txt_emb
old_emb = '        txt_emb_npy = self.get_cached_datafile("embeddings/txt_emb_species.npy")'
new_emb = '        emb_files = TOL_EMB_FILES.get(self.model_str, ("txt_emb_species.npy", "txt_emb_species.json"))\n        txt_emb_npy = self.get_cached_datafile(f"embeddings/{emb_files[0]}")'
assert old_emb in pred_src, "Could not find get_txt_emb npy line in predict.py"
pred_src = pred_src.replace(old_emb, new_emb)

# Override get_txt_names
old_names = '        txt_names_json = self.get_cached_datafile("embeddings/txt_emb_species.json")'
new_names = '        emb_files = TOL_EMB_FILES.get(self.model_str, ("txt_emb_species.npy", "txt_emb_species.json"))\n        txt_names_json = self.get_cached_datafile(f"embeddings/{emb_files[1]}")'
assert old_names in pred_src, "Could not find get_txt_names json line in predict.py"
pred_src = pred_src.replace(old_names, new_names)

pred_path.write_text(pred_src)
print("Patched predict.py: model-specific embedding filenames")
print("BioCLIP 2.5 Huge support ready")
