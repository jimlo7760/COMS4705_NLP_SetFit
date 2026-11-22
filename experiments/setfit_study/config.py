"""Single source of truth for the study protocol (mirrors the proposal).

Change a value here only if you also report the change in the milestone/final
report ("changes since the proposal").
"""

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# --- data source (author-released CSVs) and the audit numbers from the proposal ---
DATA_BASE_URL = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
                 "master/banking_data/")
RAW_FILES = {"train": "train.csv", "test": "test.csv"}
RAW_SHA256 = {  # copied from data_audit.json in the proposal folder
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
# data.py recomputes all of these and stops if any differs. Do not edit them to
# make a failing check pass: find out why the source changed and report it.
EXPECTED_COUNTS = {
    "original_train": 10003,
    "original_test": 3080,
    "label_conflicts": 0,
    "redundant_train_rows": 4,
    "cross_split_train_texts": 6,
    "clean_train": 9993,
    "n_classes": 77,
    "validation": 965,
    "pool": 9028,
    "min_pool_per_class": 32,
}

# --- seeds and sizes ---
VAL_SEED = 2026            # validation split (and the error-analysis sample)
TUNING_SEED = 0            # learning-rate selection only; never scored on test
FINAL_SEEDS = (11, 22, 33)
SHOTS = (8, 32)            # nested: the 8-shot set is a prefix of the 32-shot set
LEARNING_RATES = (1e-5, 3e-5)

# --- encoder training (CE tuning and SetFit) ---
MAX_LENGTH = 128
BATCH_SIZE = 32            # sentences for CE, pairs (= 64 sentences) for SetFit
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.10
MAX_UPDATES = {8: 300, 32: 1000}
EVAL_EVERY = {8: 50, 32: 100}
N_POS, N_NEG = 4, 4        # SetFit partners per anchor
ENCODE_BATCH_SIZE = 256    # inference only

# --- logistic regression used by Frozen+LR and SetFit ---
LR_PARAMS = dict(solver="lbfgs", C=1.0, fit_intercept=True, max_iter=2000)

# --- method names (match the report) ---
FROZEN, CE, SETFIT = "Frozen+LR", "CE tuning", "SetFit"
METHODS = (FROZEN, CE, SETFIT)
METHOD_SLUG = {FROZEN: "frozen", CE: "ce", SETFIT: "setfit"}
SLUG_METHOD = {v: k for k, v in METHOD_SLUG.items()}
