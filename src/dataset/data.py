"""File containing constants."""

# Path
PATH_DWIE = "data/DWIE/"
PATH_DWIE_DATA = PATH_DWIE + "annotated_texts"
PATH_DWIE_GROUNDTRUTH_KBS = PATH_DWIE + "groundtruth"
PATH_DWIE_INIT_KB = PATH_DWIE + "init_kb.pickle"
PATH_DWIE_TEST_FILES = PATH_DWIE + "test_files.pickle"


UNWANTED_ENT_TYPES = [
    "footer",
    "none",
    "skip",
    "time",
    "money",
    "value",
    "role",
    "religion",
    "religion-x",
]
