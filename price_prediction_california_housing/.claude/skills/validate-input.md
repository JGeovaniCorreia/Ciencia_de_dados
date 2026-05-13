---
description: Validate that a DataFrame or dict has the correct Portuguese column names for pipeline inference. Catches English/Portuguese mismatches before they cause silent errors.
---

## Usage

`/validate-input [path/to/data.csv or data.parquet]`

If no path is given, validate a single example row interactively.

## Expected columns

The pipeline requires exactly 8 input columns with Portuguese names and specific types and ranges. See **CLAUDE.md > Data Schema** for the canonical table (names, EN equivalents, types, valid ranges) and the rename snippet.

## Validation steps

1. If a file path was given, read the first 5 rows with pandas (csv or parquet).
2. Run the following checks and report each as ✅ or ❌:

   **Presence check:**
   - All 8 Portuguese columns present? (refer to CLAUDE.md > Data Schema for the exact list)
   - No unexpected extra columns? (warn but don't fail)
   - Any English column names found? (fail — print the rename snippet from CLAUDE.md)

   **Type check:**
   - All columns are numeric (float or int)?

   **Range sanity check** (use the valid ranges from CLAUDE.md > Data Schema):
   Flag any value outside the listed range.

3. If English column names are detected, print the rename snippet from CLAUDE.md > Data Schema and instruct the user to apply it before calling predict.

4. If no file was given, generate a minimal valid example row using the column names and mid-range values from CLAUDE.md > Data Schema.

## Final verdict

Print either:
> ✅ Input is valid — ready for `pipeline.predict(X)`
or
> ❌ Validation failed — fix the issues above before calling predict.
