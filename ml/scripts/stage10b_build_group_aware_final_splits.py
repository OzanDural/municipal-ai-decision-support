import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# INPUTS
# ============================================================

FINAL_DATASET = Path(
    "data/processed/master_final.jsonl"
)

PROVISIONAL_TRAIN = Path(
    "data/splits/provisional/train.jsonl"
)

PROVISIONAL_VAL = Path(
    "data/splits/provisional/validation.jsonl"
)

PROVISIONAL_TEST = Path(
    "data/splits/provisional/test.jsonl"
)

SEMANTIC_CANDIDATES = Path(
    "ml/reports/dataset_qa/"
    "06b_semantic_duplicate_candidates.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

FINAL_SPLIT_DIR = Path(
    "data/splits/final"
)

REPORT_DIR = Path(
    "ml/reports/dataset_qa"
)

FINAL_TRAIN = (
    FINAL_SPLIT_DIR / "train.jsonl"
)

FINAL_VAL = (
    FINAL_SPLIT_DIR / "validation.jsonl"
)

FINAL_TEST = (
    FINAL_SPLIT_DIR / "test.jsonl"
)

MANIFEST_OUTPUT = REPORT_DIR / (
    "10b_final_group_aware_split_manifest.csv"
)

GROUP_OUTPUT = REPORT_DIR / (
    "10b_semantic_split_groups.csv"
)

SUMMARY_OUTPUT = REPORT_DIR / (
    "10b_final_group_aware_split_summary.json"
)


# ============================================================
# EXPECTED HASHES
# ============================================================

EXPECTED_FINAL_SHA256 = (
    "c6d8a06e0fb757c3b1db11a87959f7086fd4b22da5c97268522c5fd4d77ed1c7"
)

EXPECTED_PROVISIONAL_TRAIN_SHA256 = (
    "1cc1c03db3e25a8a631267d6d16b712da9ae13059935048009c57371509ca8c4"
)

EXPECTED_PROVISIONAL_VAL_SHA256 = (
    "20f7f2f0ae0652079cc82f2dc5841b0890af2a2a7636154990a3b381d87648cb"
)

EXPECTED_PROVISIONAL_TEST_SHA256 = (
    "df52121b604e3934891a60df92d24c40b28c8549ec020d35ea23c9c4889baefd"
)


EXPECTED_FINAL_COUNT = 1667

SPLIT_NAMES = [
    "TRAIN",
    "VALIDATION",
    "TEST",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path):

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


def stratum_key(record):

    return (
        record["category"]
        + "||"
        + record["scenario"]
    )


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, items):

        self.parent = {
            item: item
            for item in items
        }

        self.rank = {
            item: 0
            for item in items
        }


    def find(self, item):

        if (
            self.parent[item]
            != item
        ):

            self.parent[item] = (
                self.find(
                    self.parent[item]
                )
            )

        return self.parent[item]


    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):

            root_a, root_b = (
                root_b,
                root_a,
            )

        self.parent[
            root_b
        ] = root_a

        if (
            self.rank[root_a]
            == self.rank[root_b]
        ):

            self.rank[
                root_a
            ] += 1


# ============================================================
# VERIFY INPUT HASHES
# ============================================================

hash_checks = [
    (
        "FINAL",
        FINAL_DATASET,
        EXPECTED_FINAL_SHA256,
    ),
    (
        "PROVISIONAL_TRAIN",
        PROVISIONAL_TRAIN,
        EXPECTED_PROVISIONAL_TRAIN_SHA256,
    ),
    (
        "PROVISIONAL_VALIDATION",
        PROVISIONAL_VAL,
        EXPECTED_PROVISIONAL_VAL_SHA256,
    ),
    (
        "PROVISIONAL_TEST",
        PROVISIONAL_TEST,
        EXPECTED_PROVISIONAL_TEST_SHA256,
    ),
]


for (
    name,
    path,
    expected,
) in hash_checks:

    actual = sha256_file(
        path
    )

    if actual != expected:

        raise ValueError(
            f"{name} hash mismatch!\n"
            f"Expected: {expected}\n"
            f"Actual  : {actual}"
        )


# ============================================================
# LOAD DATA
# ============================================================

records = load_jsonl(
    FINAL_DATASET
)

train_records = load_jsonl(
    PROVISIONAL_TRAIN
)

val_records = load_jsonl(
    PROVISIONAL_VAL
)

test_records = load_jsonl(
    PROVISIONAL_TEST
)


if len(records) != 1667:

    raise ValueError(
        f"Final dataset expected 1667, "
        f"found {len(records)}."
    )


record_map = {
    record["sample_id"]:
        record
    for record in records
}


if len(record_map) != 1667:

    raise ValueError(
        "Duplicate sample_id in final dataset."
    )


# ============================================================
# PROVISIONAL SPLIT MAP
# ============================================================

old_split_map = {}


for split_name, split_records in [
    (
        "TRAIN",
        train_records,
    ),
    (
        "VALIDATION",
        val_records,
    ),
    (
        "TEST",
        test_records,
    ),
]:

    for record in split_records:

        sid = record[
            "sample_id"
        ]

        if sid in old_split_map:

            raise ValueError(
                f"Duplicate split assignment: {sid}"
            )

        old_split_map[
            sid
        ] = split_name


if set(old_split_map) != set(
    record_map
):

    raise ValueError(
        "Provisional splits do not cover "
        "the final dataset exactly."
    )


# ============================================================
# ORIGINAL TARGET COUNTS PER STRATUM
#
# Group-aware reconstruction will preserve these EXACTLY.
# ============================================================

target_counts = defaultdict(
    Counter
)


for sid, split_name in (
    old_split_map.items()
):

    key = stratum_key(
        record_map[sid]
    )

    target_counts[
        key
    ][
        split_name
    ] += 1


# ============================================================
# READ STAGE 6B DUPLICATE CANDIDATE GRAPH
# ============================================================

candidate_rows = []


with SEMANTIC_CANDIDATES.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:

    reader = csv.DictReader(f)

    if not reader.fieldnames:

        raise ValueError(
            "Semantic candidate CSV "
            "header missing."
        )

    required = {
        "sample_id_1",
        "sample_id_2",
    }

    missing = (
        required
        - set(reader.fieldnames)
    )

    if missing:

        raise ValueError(
            f"Missing fields: {missing}"
        )

    candidate_rows = list(
        reader
    )


if len(candidate_rows) != 305:

    raise ValueError(
        f"Expected 305 Stage6B candidate "
        f"edges, found {len(candidate_rows)}."
    )


# ============================================================
# BUILD SURVIVOR GRAPH
# ============================================================

uf = UnionFind(
    record_map.keys()
)


survivor_edges = []


for row in candidate_rows:

    id_1 = row[
        "sample_id_1"
    ]

    id_2 = row[
        "sample_id_2"
    ]


    # Stage6B was created on the 1750-record master.
    # Ignore records removed during QA.
    if (
        id_1 not in record_map
        or id_2 not in record_map
    ):

        continue


    record_1 = record_map[
        id_1
    ]

    record_2 = record_map[
        id_2
    ]


    # The Stage6B duplicate pool must stay inside
    # category + scenario strata.
    if (
        record_1["category"]
        != record_2["category"]
        or
        record_1["scenario"]
        != record_2["scenario"]
    ):

        raise ValueError(
            "Duplicate candidate crosses "
            "category/scenario strata:\n"
            f"{id_1} <-> {id_2}"
        )


    survivor_edges.append(
        (
            id_1,
            id_2,
        )
    )

    uf.union(
        id_1,
        id_2,
    )


if len(survivor_edges) != 139:

    raise ValueError(
        f"Expected 139 Stage6B edges "
        f"with both final survivors, "
        f"found {len(survivor_edges)}."
    )


# ============================================================
# COMPONENTS
# ============================================================

components_by_root = defaultdict(
    list
)


for sid in record_map:

    root = uf.find(
        sid
    )

    components_by_root[
        root
    ].append(
        sid
    )


components = []


for members in (
    components_by_root.values()
):

    members = sorted(
        members
    )

    strata = {
        stratum_key(
            record_map[sid]
        )
        for sid in members
    }


    if len(strata) != 1:

        raise ValueError(
            "Semantic component spans "
            "multiple category/scenario strata."
        )


    components.append({
        "members":
            members,

        "size":
            len(members),

        "stratum":
            next(
                iter(strata)
            ),
    })


components.sort(
    key=lambda c: (
        c["stratum"],
        c["members"][0],
    )
)


# ============================================================
# GROUP COMPONENTS BY STRATUM
# ============================================================

stratum_components = defaultdict(
    list
)


for component in components:

    stratum_components[
        component["stratum"]
    ].append(
        component
    )


if len(
    stratum_components
) != 25:

    raise ValueError(
        f"Expected 25 strata, "
        f"found {len(stratum_components)}."
    )


# ============================================================
# OPTIMAL PER-STRATUM GROUP ASSIGNMENT
#
# Dynamic programming:
#
# - preserve exact TRAIN / VAL / TEST counts
#   from Stage 9 for each category+scenario stratum
#
# - every semantic component is indivisible
#
# - maximize number of records that remain in
#   their old Stage 9 split
#
# This minimizes disruption while eliminating
# candidate-edge leakage.
# ============================================================

new_split_map = {}

assignment_records_kept = 0


for stratum in sorted(
    stratum_components
):

    comps = (
        stratum_components[
            stratum
        ]
    )

    target_train = (
        target_counts[
            stratum
        ][
            "TRAIN"
        ]
    )

    target_val = (
        target_counts[
            stratum
        ][
            "VALIDATION"
        ]
    )

    target_test = (
        target_counts[
            stratum
        ][
            "TEST"
        ]
    )


    total_target = (
        target_train
        + target_val
        + target_test
    )

    total_components = sum(
        comp["size"]
        for comp in comps
    )


    if (
        total_target
        != total_components
    ):

        raise ValueError(
            f"Stratum count mismatch: "
            f"{stratum}"
        )


    # state:
    # (train_count, val_count)
    #
    # value:
    # (kept_score, choices_tuple)
    #
    # TEST count is implicit.
    dp = {
        (0, 0):
            (
                0,
                tuple(),
            )
    }


    processed = 0


    for comp in comps:

        size = comp[
            "size"
        ]

        keep_scores = {}


        for split_name in (
            SPLIT_NAMES
        ):

            keep_scores[
                split_name
            ] = sum(
                1
                for sid
                in comp["members"]
                if old_split_map[
                    sid
                ] == split_name
            )


        next_dp = {}


        for (
            train_count,
            val_count,
        ), (
            current_score,
            choices,
        ) in dp.items():

            current_test = (
                processed
                - train_count
                - val_count
            )


            for split_name in (
                SPLIT_NAMES
            ):

                new_train = (
                    train_count
                )

                new_val = (
                    val_count
                )

                new_test = (
                    current_test
                )


                if (
                    split_name
                    == "TRAIN"
                ):

                    new_train += size

                elif (
                    split_name
                    == "VALIDATION"
                ):

                    new_val += size

                else:

                    new_test += size


                if (
                    new_train
                    > target_train
                    or new_val
                    > target_val
                    or new_test
                    > target_test
                ):

                    continue


                state = (
                    new_train,
                    new_val,
                )


                score = (
                    current_score
                    + keep_scores[
                        split_name
                    ]
                )


                new_choices = (
                    choices
                    + (
                        split_name,
                    )
                )


                existing = (
                    next_dp.get(
                        state
                    )
                )


                if (
                    existing is None
                    or score
                    > existing[0]
                ):

                    next_dp[
                        state
                    ] = (
                        score,
                        new_choices,
                    )


        dp = next_dp

        processed += size


        if not dp:

            raise ValueError(
                "No feasible group-aware "
                f"assignment for stratum "
                f"{stratum}."
            )


    final_state = (
        target_train,
        target_val,
    )


    if final_state not in dp:

        raise ValueError(
            "Could not preserve exact split "
            f"counts for stratum {stratum}."
        )


    best_score, choices = (
        dp[
            final_state
        ]
    )


    assignment_records_kept += (
        best_score
    )


    if (
        len(choices)
        != len(comps)
    ):

        raise ValueError(
            "DP assignment length error."
        )


    for comp, split_name in zip(
        comps,
        choices,
    ):

        for sid in comp[
            "members"
        ]:

            if sid in new_split_map:

                raise ValueError(
                    f"Duplicate final assignment: "
                    f"{sid}"
                )

            new_split_map[
                sid
            ] = split_name


# ============================================================
# FINAL ASSIGNMENT INTEGRITY
# ============================================================

if set(new_split_map) != set(
    record_map
):

    raise ValueError(
        "Final group-aware split does "
        "not cover all records."
    )


# ============================================================
# VERIFY EXACT PER-STRATUM COUNTS PRESERVED
# ============================================================

new_target_counts = defaultdict(
    Counter
)


for sid, split_name in (
    new_split_map.items()
):

    key = stratum_key(
        record_map[
            sid
        ]
    )

    new_target_counts[
        key
    ][
        split_name
    ] += 1


for stratum in target_counts:

    if (
        target_counts[
            stratum
        ]
        != new_target_counts[
            stratum
        ]
    ):

        raise ValueError(
            "Stratum distribution changed:\n"
            f"{stratum}\n"
            f"Old: "
            f"{dict(target_counts[stratum])}\n"
            f"New: "
            f"{dict(new_target_counts[stratum])}"
        )


# ============================================================
# CROSS-SPLIT EDGE CHECK
# ============================================================

cross_split_before = []

cross_split_after = []


for id_1, id_2 in (
    survivor_edges
):

    if (
        old_split_map[
            id_1
        ]
        != old_split_map[
            id_2
        ]
    ):

        cross_split_before.append(
            (
                id_1,
                id_2,
            )
        )


    if (
        new_split_map[
            id_1
        ]
        != new_split_map[
            id_2
        ]
    ):

        cross_split_after.append(
            (
                id_1,
                id_2,
            )
        )


if len(
    cross_split_before
) != 62:

    raise ValueError(
        "Expected 62 provisional "
        f"cross-split edges, found "
        f"{len(cross_split_before)}."
    )


if cross_split_after:

    raise ValueError(
        "Group-aware split still contains "
        f"{len(cross_split_after)} "
        "cross-split duplicate-candidate edges."
    )


# ============================================================
# MOVEMENT AUDIT
# ============================================================

moved_ids = [
    sid
    for sid in record_map
    if (
        old_split_map[
            sid
        ]
        != new_split_map[
            sid
        ]
    )
]


movement_counts = Counter()


for sid in moved_ids:

    movement_counts[
        (
            old_split_map[sid]
            + "_TO_"
            + new_split_map[sid]
        )
    ] += 1


# ============================================================
# BUILD FINAL SPLITS
# ============================================================

final_train_records = []
final_val_records = []
final_test_records = []


for record in records:

    sid = record[
        "sample_id"
    ]

    split_name = (
        new_split_map[
            sid
        ]
    )


    if (
        split_name
        == "TRAIN"
    ):

        final_train_records.append(
            record
        )

    elif (
        split_name
        == "VALIDATION"
    ):

        final_val_records.append(
            record
        )

    elif (
        split_name
        == "TEST"
    ):

        final_test_records.append(
            record
        )

    else:

        raise ValueError(
            f"Unknown split: "
            f"{split_name}"
        )


if len(
    final_train_records
) != 1167:

    raise ValueError(
        "Final TRAIN size != 1167"
    )


if len(
    final_val_records
) != 250:

    raise ValueError(
        "Final VALIDATION size != 250"
    )


if len(
    final_test_records
) != 250:

    raise ValueError(
        "Final TEST size != 250"
    )


# ============================================================
# ID OVERLAP / CONSERVATION
# ============================================================

train_ids = {
    r["sample_id"]
    for r in final_train_records
}

val_ids = {
    r["sample_id"]
    for r in final_val_records
}

test_ids = {
    r["sample_id"]
    for r in final_test_records
}


if train_ids & val_ids:

    raise ValueError(
        "TRAIN/VALIDATION overlap."
    )


if train_ids & test_ids:

    raise ValueError(
        "TRAIN/TEST overlap."
    )


if val_ids & test_ids:

    raise ValueError(
        "VALIDATION/TEST overlap."
    )


if (
    train_ids
    | val_ids
    | test_ids
) != set(
    record_map
):

    raise ValueError(
        "Final split conservation failed."
    )


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FINAL_SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# WRITE FINAL SPLITS
# ============================================================

for path, split_records in [
    (
        FINAL_TRAIN,
        final_train_records,
    ),
    (
        FINAL_VAL,
        final_val_records,
    ),
    (
        FINAL_TEST,
        final_test_records,
    ),
]:

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in split_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# COMPONENT / GROUP IDS
# ============================================================

component_id_map = {}

component_size_map = {}


non_singleton_components = 0
largest_component = 0


for index, component in enumerate(
    components,
    start=1,
):

    group_id = (
        f"SEMGRP_{index:04d}"
    )

    size = component[
        "size"
    ]

    if size > 1:

        non_singleton_components += 1

    largest_component = max(
        largest_component,
        size,
    )


    for sid in component[
        "members"
    ]:

        component_id_map[
            sid
        ] = group_id

        component_size_map[
            sid
        ] = size


# ============================================================
# MANIFEST
# ============================================================

manifest_rows = []


for record in records:

    sid = record[
        "sample_id"
    ]

    manifest_rows.append({
        "sample_id":
            sid,

        "category":
            record["category"],

        "scenario":
            record["scenario"],

        "semantic_group_id":
            component_id_map[
                sid
            ],

        "semantic_group_size":
            component_size_map[
                sid
            ],

        "provisional_split":
            old_split_map[
                sid
            ],

        "final_split":
            new_split_map[
                sid
            ],

        "moved":
            (
                old_split_map[
                    sid
                ]
                != new_split_map[
                    sid
                ]
            ),
    })


manifest_fields = [
    "sample_id",
    "category",
    "scenario",
    "semantic_group_id",
    "semantic_group_size",
    "provisional_split",
    "final_split",
    "moved",
]


with MANIFEST_OUTPUT.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=manifest_fields,
    )

    writer.writeheader()

    writer.writerows(
        manifest_rows
    )


# ============================================================
# GROUP AUDIT
# ============================================================

group_rows = []


for index, component in enumerate(
    components,
    start=1,
):

    if component[
        "size"
    ] <= 1:

        continue


    group_id = (
        f"SEMGRP_{index:04d}"
    )

    group_split = (
        new_split_map[
            component[
                "members"
            ][0]
        ]
    )


    if len({
        new_split_map[sid]
        for sid
        in component["members"]
    }) != 1:

        raise ValueError(
            "Semantic group spans "
            "multiple final splits."
        )


    for sid in component[
        "members"
    ]:

        group_rows.append({
            "semantic_group_id":
                group_id,

            "group_size":
                component["size"],

            "sample_id":
                sid,

            "category":
                record_map[
                    sid
                ][
                    "category"
                ],

            "scenario":
                record_map[
                    sid
                ][
                    "scenario"
                ],

            "provisional_split":
                old_split_map[
                    sid
                ],

            "final_split":
                group_split,

            "moved":
                (
                    old_split_map[
                        sid
                    ]
                    != group_split
                ),
        })


group_fields = [
    "semantic_group_id",
    "group_size",
    "sample_id",
    "category",
    "scenario",
    "provisional_split",
    "final_split",
    "moved",
]


with GROUP_OUTPUT.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=group_fields,
    )

    writer.writeheader()

    writer.writerows(
        group_rows
    )


# ============================================================
# FINAL DISTRIBUTIONS
# ============================================================

def category_counts(
    split_records
):

    return dict(
        sorted(
            Counter(
                r["category"]
                for r
                in split_records
            ).items()
        )
    )


def stratum_counts(
    split_records
):

    return dict(
        sorted(
            Counter(
                stratum_key(r)
                for r
                in split_records
            ).items()
        )
    )


provisional_category = {
    "TRAIN":
        category_counts(
            train_records
        ),

    "VALIDATION":
        category_counts(
            val_records
        ),

    "TEST":
        category_counts(
            test_records
        ),
}


final_category = {
    "TRAIN":
        category_counts(
            final_train_records
        ),

    "VALIDATION":
        category_counts(
            final_val_records
        ),

    "TEST":
        category_counts(
            final_test_records
        ),
}


if (
    provisional_category
    != final_category
):

    raise ValueError(
        "Category distributions changed."
    )


provisional_strata = {
    "TRAIN":
        stratum_counts(
            train_records
        ),

    "VALIDATION":
        stratum_counts(
            val_records
        ),

    "TEST":
        stratum_counts(
            test_records
        ),
}


final_strata = {
    "TRAIN":
        stratum_counts(
            final_train_records
        ),

    "VALIDATION":
        stratum_counts(
            final_val_records
        ),

    "TEST":
        stratum_counts(
            final_test_records
        ),
}


if (
    provisional_strata
    != final_strata
):

    raise ValueError(
        "Category+scenario distributions changed."
    )


# ============================================================
# FINAL HASHES
# ============================================================

train_hash = sha256_file(
    FINAL_TRAIN
)

val_hash = sha256_file(
    FINAL_VAL
)

test_hash = sha256_file(
    FINAL_TEST
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "status":
        "FINAL_GROUP_AWARE_SPLIT_FROZEN",

    "input_final_dataset_count":
        len(records),

    "input_final_dataset_sha256":
        sha256_file(
            FINAL_DATASET
        ),

    "split_method":
        (
            "CATEGORY_SCENARIO_STRATIFIED_"
            "SEMANTIC_GROUP_AWARE"
        ),

    "semantic_group_source":
        (
            "Stage6B final-survivor "
            "duplicate-candidate graph"
        ),

    "stage6b_global_candidate_edges":
        len(
            candidate_rows
        ),

    "stage6b_final_survivor_edges":
        len(
            survivor_edges
        ),

    "cross_split_candidate_edges_before":
        len(
            cross_split_before
        ),

    "cross_split_candidate_edges_after":
        len(
            cross_split_after
        ),

    "semantic_component_count":
        len(
            components
        ),

    "non_singleton_semantic_components":
        non_singleton_components,

    "largest_semantic_component":
        largest_component,

    "moved_record_count":
        len(
            moved_ids
        ),

    "records_kept_in_original_split":
        assignment_records_kept,

    "movement_distribution":
        dict(
            sorted(
                movement_counts.items()
            )
        ),

    "final_train_count":
        len(
            final_train_records
        ),

    "final_validation_count":
        len(
            final_val_records
        ),

    "final_test_count":
        len(
            final_test_records
        ),

    "category_distribution_preserved_exactly":
        True,

    "category_scenario_distribution_preserved_exactly":
        True,

    "split_overlap": {
        "train_validation":
            len(
                train_ids
                & val_ids
            ),

        "train_test":
            len(
                train_ids
                & test_ids
            ),

        "validation_test":
            len(
                val_ids
                & test_ids
            ),
    },

    "final_hashes": {
        "train":
            train_hash,

        "validation":
            val_hash,

        "test":
            test_hash,
    },

    "test_frozen":
        True,

    "frozen_test_sha256":
        test_hash,

    "dataset_modified":
        False,

    "records_removed":
        0,

    "policy": (
        "No additional records were rejected. "
        "The Stage6B final-survivor semantic "
        "near-duplicate graph was treated as "
        "an indivisible grouping constraint. "
        "Dynamic programming reassigned complete "
        "semantic components while preserving "
        "the exact Stage9 category+scenario counts "
        "for TRAIN, VALIDATION and TEST and "
        "minimizing changes from the provisional "
        "split. All Stage6B survivor candidate "
        "edges are now internal to one split."
    ),
}


with SUMMARY_OUTPUT.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# TERMINAL
# ============================================================

print("=" * 78)
print(
    "STAGE 10B - GROUP-AWARE "
    "FINAL SPLIT & LEAKAGE CLOSURE"
)
print("=" * 78)

print(
    f"Final dataset               : "
    f"{len(records)}"
)

print(
    f"Stage6B candidate edges     : "
    f"{len(candidate_rows)}"
)

print(
    f"Both-final-survivor edges   : "
    f"{len(survivor_edges)}"
)


print("\nLeakage candidate edges:")

print(
    f"  BEFORE group-aware split  : "
    f"{len(cross_split_before)}"
)

print(
    f"  AFTER group-aware split   : "
    f"{len(cross_split_after)}"
)


print("\nSemantic groups:")

print(
    f"  Total components          : "
    f"{len(components)}"
)

print(
    f"  Non-singleton components  : "
    f"{non_singleton_components}"
)

print(
    f"  Largest component         : "
    f"{largest_component}"
)


print("\nSplit movement:")

print(
    f"  Records moved             : "
    f"{len(moved_ids)}"
)

print(
    f"  Records unchanged         : "
    f"{assignment_records_kept}"
)

for key, value in sorted(
    movement_counts.items()
):

    print(
        f"  {key:<26}: {value}"
    )


print("\nFinal split sizes:")

print(
    f"  TRAIN                     : "
    f"{len(final_train_records)}"
)

print(
    f"  VALIDATION                : "
    f"{len(final_val_records)}"
)

print(
    f"  TEST                      : "
    f"{len(final_test_records)}"
)


print("\nDistribution integrity:")

print(
    "  Category                  : "
    "PRESERVED EXACTLY"
)

print(
    "  Category + scenario       : "
    "PRESERVED EXACTLY"
)

print(
    f"  TRAIN / VAL overlap       : "
    f"{len(train_ids & val_ids)}"
)

print(
    f"  TRAIN / TEST overlap      : "
    f"{len(train_ids & test_ids)}"
)

print(
    f"  VAL / TEST overlap        : "
    f"{len(val_ids & test_ids)}"
)


print("\nFINAL HASHES:")

print(
    f"  TRAIN                     : "
    f"{train_hash}"
)

print(
    f"  VALIDATION                : "
    f"{val_hash}"
)

print(
    f"  TEST                      : "
    f"{test_hash}"
)


print("\nOutputs:")

print(
    f"  Train    : {FINAL_TRAIN}"
)

print(
    f"  Val      : {FINAL_VAL}"
)

print(
    f"  Test     : {FINAL_TEST}"
)

print(
    f"  Manifest : {MANIFEST_OUTPUT}"
)

print(
    f"  Groups   : {GROUP_OUTPUT}"
)

print(
    f"  Summary  : {SUMMARY_OUTPUT}"
)


print("\nTEST FROZEN: True")

print(
    "STATUS: "
    "FINAL_GROUP_AWARE_SPLIT_FROZEN"
)