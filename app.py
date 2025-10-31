# app.py
import streamlit as st
import pandas as pd
import math
import io
import logging
import traceback

# ---------- Logging ----------
logger = logging.getLogger("assignment2_alloc")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler("assignment2_app.log", mode="a", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

st.set_page_config(page_title="BTP/MTP Allocation (faculty-columns-as-ranks)", layout="wide")
st.title("BTP/MTP Allocation — Grouped allocation by CGPA + ranked faculty-columns")
st.markdown("""
Upload the input CSV/XLSX file that has columns:
`Roll, Name, Email, CGPA, <Faculty1>, <Faculty2>, ...`  
Where each faculty column contains a numeric rank indicating student's preference for that faculty (1 = 1st choice).
The app will produce:
- `output_btp_mtp_allocation.csv` (Roll, Name, Email, CGPA, Allocated)
- `fac_preference_count.csv` (Fac, Count Pref 1, Count Pref 2, ..., Count Pref n)
""")

# ---------- Core allocation logic ----------
def read_input(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".xlsx") or uploaded_file.name.lower().endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        logger.exception("Failed to read uploaded file.")
        raise

def detect_faculty_columns(df):
    """Return list of faculty columns (all columns after 'CGPA')"""
    if 'CGPA' not in df.columns:
        raise ValueError("Input must have a column named 'CGPA' (case-sensitive).")
    cols = list(df.columns)
    cgpa_idx = cols.index('CGPA')
    fac_cols = cols[cgpa_idx + 1:]
    if len(fac_cols) == 0:
        raise ValueError("No faculty columns found after 'CGPA'.")
    return fac_cols

def build_student_prefs_from_rank_columns(row, fac_cols):
    """
    Given a row and list of faculty columns (where values are numeric ranks),
    return list of faculty names ordered by ascending rank (1,2,3,...).
    If some faculty rank is NaN or not positive integer, that faculty is ignored.
    """
    ranks = []
    for f in fac_cols:
        val = row.get(f)
        try:
            if pd.isna(val):
                continue
            # convert to int if possible
            r = int(float(val))
            if r <= 0:
                continue
            ranks.append((r, f))
        except Exception:
            # not numeric - ignore
            continue
    # sort by rank ascending (1 first)
    ranks.sort(key=lambda x: x[0])
    prefs = [f for _, f in ranks]
    return prefs

def compute_fac_pref_counts(df, fac_cols):
    """
    Build a DataFrame with columns: Fac, Count Pref 1, Count Pref 2, ..., Count Pref n
    Count Pref k indicates how many students ranked that faculty at position k.
    """
    # For each student, build their ordered pref list (by numeric rank)
    all_prefs = []
    for _, row in df.iterrows():
        prefs = build_student_prefs_from_rank_columns(row, fac_cols)
        all_prefs.append(prefs)
    S = len(all_prefs)
    # find maximum preference length across students (should be n)
    max_pref_len = max((len(p) for p in all_prefs), default=0)
    # initialize counts
    fac_list = list(fac_cols)
    counts = {fac: [0]*max_pref_len for fac in fac_list}
    for prefs in all_prefs:
        for pos, fac in enumerate(prefs):
            counts[fac][pos] += 1
    # build DataFrame
    rows = []
    for fac in fac_list:
        row = {'Fac': fac}
        for k in range(max_pref_len):
            row[f'Count Pref {k+1}'] = counts[fac][k]
        rows.append(row)
    fac_stats_df = pd.DataFrame(rows)
    return fac_stats_df

def allocate_grouped_by_CGPA_rank_matrix(df, fac_cols):
    """
    Implements the grouped allocation exactly as you specified:
    - S = total students, n = number of faculty (len(fac_cols))
    - groups = ceil(S / n). Partition the students sorted by CGPA descending into groups.
    - For each group, each faculty can be assigned to at most one student in that group.
    - For each student in group (CGPA order), assign highest-preference faculty that is still free in that group.
      If none of student's preferred faculties are free, assign any remaining free faculty (deterministic by fac_cols order).
    Returns allocation_df (Roll, Name, Email, CGPA, Allocated, GroupNo) and fac_stats_df (constructed separately).
    """
    try:
        # prepare sorted students
        df = df.copy()
        df['CGPA_num'] = pd.to_numeric(df['CGPA'], errors='coerce')
        df_sorted = df.sort_values(by='CGPA_num', ascending=False).reset_index(drop=True)
        S = len(df_sorted)
        n = len(fac_cols)
        groups = math.ceil(S / n)

        allocation_rows = []
        faculty_total_alloc = {f: 0 for f in fac_cols}

        for g in range(groups):
            start = g * n
            end = min(start + n, S)
            group_df = df_sorted.iloc[start:end].reset_index(drop=True)
            free_faculties = set(fac_cols)  # all faculties available for each group initially
            # For each student in this group in CGPA order
            for idx, row in group_df.iterrows():
                roll = row.get('Roll', '')
                name = row.get('Name', '')
                email = row.get('Email', '')
                cgpa = row.get('CGPA')
                # student's prefs based on numeric ranks in faculty columns
                prefs = build_student_prefs_from_rank_columns(row, fac_cols)
                assigned = None
                for p in prefs:
                    if p in free_faculties:
                        assigned = p
                        break
                if assigned is None:
                    # assign any remaining free faculty in deterministic order
                    if free_faculties:
                        for f in fac_cols:
                            if f in free_faculties:
                                assigned = f
                                break
                    else:
                        assigned = "UNASSIGNED"
                # mark assigned faculty used in this group
                if assigned in free_faculties:
                    free_faculties.remove(assigned)
                faculty_total_alloc.setdefault(assigned, 0)
                faculty_total_alloc[assigned] += 1
                allocation_rows.append({
                    'Roll': roll,
                    'Name': name,
                    'Email': email,
                    'CGPA': cgpa,
                    'Allocated': assigned,
                    'GroupNo': g+1
                })

        allocation_df = pd.DataFrame(allocation_rows)
        return allocation_df, faculty_total_alloc, groups
    except Exception:
        logger.exception("Allocation failed")
        raise

# ---------- Streamlit UI ----------
uploaded = st.file_uploader("Upload input file (.csv or .xlsx) — faculty columns must be numeric ranks", type=['csv','xlsx','xls'])
if not uploaded:
    st.info("Upload your input file (CSV or Excel). Columns should be: Roll, Name, Email, CGPA, <faculty columns containing numeric ranks 1..n>.")
    st.stop()

try:
    df_input = read_input(uploaded)
except Exception as e:
    st.error(f"Failed to read input file: {e}")
    st.stop()

# show preview
st.subheader("Input preview (first 20 rows)")
st.dataframe(df_input.head(20))

# detect faculty columns
try:
    fac_cols = detect_faculty_columns(df_input)
    st.write(f"Detected {len(fac_cols)} faculty columns.")
except Exception as e:
    st.error(str(e))
    st.stop()

# compute faculty preference counts table
try:
    fac_pref_counts_df = compute_fac_pref_counts(df_input, fac_cols)
except Exception:
    st.error("Failed to compute faculty preference counts.")
    logger.exception("Failed computing fac pref counts")
    st.stop()

# run allocation
try:
    with st.spinner("Running grouped allocation (ceil(S/n)) ..."):
        alloc_df, faculty_total_alloc, groups = allocate_grouped_by_CGPA_rank_matrix(df_input, fac_cols)
    st.success("Allocation complete.")
    st.write(f"Total students: **{len(df_input)}**, Faculties: **{len(fac_cols)}**, Groups: **{groups}**")
    st.subheader("Allocation preview (first 50 rows)")
    st.dataframe(alloc_df.head(50))

    # create allocation CSV in expected format: Roll, Name, Email, CGPA, Allocated
    alloc_out_df = alloc_df[['Roll','Name','Email','CGPA','Allocated']].copy()

    # Prepare faculty pref counts CSV in expected shape:
    # fac_pref_counts_df already has columns: Fac, Count Pref 1..Count Pref m
    # ensure all Count Pref columns up to n exist (if some students had fewer prefs, fill with zeros)
    max_pref_cols = max([int(c.split()[-1]) for c in fac_pref_counts_df.columns if c.startswith('Count Pref')] + [0])
    # Download buttons
    buf_alloc = io.StringIO()
    alloc_out_df.to_csv(buf_alloc, index=False)
    alloc_csv = buf_alloc.getvalue()
    st.download_button("Download allocation CSV (output_btp_mtp_allocation.csv)", data=alloc_csv,
                       file_name="output_btp_mtp_allocation.csv", mime="text/csv")

    buf_stats = io.StringIO()
    fac_pref_counts_df.to_csv(buf_stats, index=False)
    stats_csv = buf_stats.getvalue()
    st.download_button("Download faculty pref counts (fac_preference_count.csv)", data=stats_csv,
                       file_name="fac_preference_count.csv", mime="text/csv")

    st.subheader("Faculty preference counts (how many students gave each pref position)")
    st.dataframe(fac_pref_counts_df)

    st.info("Log file: assignment2_app.log (app folder).")

except Exception as e:
    st.error(f"Allocation failed: {e}")
    logger.exception("Allocation failed during Streamlit run.")