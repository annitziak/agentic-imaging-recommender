import pdfplumber
import re
import json

def is_table_header(line):
    return "Procedure Appropriateness Category" in line

def is_variant_start(line):
    return re.match(r"^Variant \d:", line.strip())

def is_section_heading(line):
    return "Discussion" in line or line.strip() == ""

def is_radiation_level(line):
    return line.strip() in ['O', '☢', '☢☢', '☢☢☢', '☢☢☢☢', '☢☢☢☢☢']

def parse_table_rows(table_lines):
    """
    Parses the rows of the procedure appropriateness table.
    """
    output = []
    header_found = False
    for line in table_lines:
        if not header_found:
            if "Procedure Appropriateness Category" in line:
                header_found = True
            continue
        if is_radiation_level(line) or not line.strip():
            continue
        m = re.search(r'(Usually|May|Not)\s+[A-Z]', line)
        if m:
            split_idx = m.start()
            proc = line[:split_idx].strip()
            appr = line[split_idx:].strip()
            output.append({"procedure": proc, "appropriateness": appr})
        else:
            continue
    return output

# --- MAIN LOGIC ---
input_path = "" # enter your pdf path
with pdfplumber.open(input_path) as pdf:
    lines = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            lines.extend(text.split('\n'))

# get the clinical category from the first page e.g. "Breast Pain"
current_category = lines[3].strip() if len(lines) > 3 else ""

# --- FIND SECTION INDICES ---
end_point_tables = None
start_review_index = None
end_review_index = None
start_summary_recommendations_index = None
start_summary_evidence_index = None
start_appropriateness_category_index = None
start_reference_index = None


# Note: the names of the sections sometimes change, depending on the year of publication. You might need to change. 
for i, line in enumerate(lines):
    if current_category and (line.strip().lower() == current_category.lower()):
        end_point_tables = i
    if "Summary of Literature Review" in line and start_review_index is None:
        start_review_index = i + 1
    if "Discussion of Procedures by Variant" in line and start_review_index is not None and end_review_index is None:
        end_review_index = i
    if "Summary of Recommendations" in line and start_summary_recommendations_index is None:
        start_summary_recommendations_index = i
    if "Summary of Evidence" or "Discussion of Procedures by Variant" in line and start_summary_evidence_index is None:
        start_summary_evidence_index = i
    if "Appropriateness Category Names and Definitions" in line and start_appropriateness_category_index is None:
        start_appropriateness_category_index = i
    if "Reference" in line and start_reference_index is None: #or references
        print(i)
        start_reference_index = i+1

##print("End point tables:", end_point_tables)
##print("Start review index:", start_review_index)
##print("End review index:", end_review_index)
##print("Start summary recommendations index:", start_summary_recommendations_index)
##print("Start summary evidence index:", start_summary_evidence_index)
##print("Start appropriateness category index:", start_appropriateness_category_index)
##print("Start reference index:", start_reference_index)


# --- SLICE SECTIONS ---
table_lines = lines[:end_point_tables] if end_point_tables else []
introduction_lines = lines[start_review_index:end_review_index] if (start_review_index is not None and end_review_index is not None) else []
summary_recommendations_lines = lines[start_summary_recommendations_index:start_summary_evidence_index] if (start_summary_recommendations_index is not None and start_summary_evidence_index is not None) else []
summary_evidence_lines = lines[start_summary_evidence_index:start_appropriateness_category_index] if (start_summary_evidence_index is not None and start_appropriateness_category_index is not None) else []
references_lines = lines[start_reference_index:-1] if start_reference_index is not None else []

# --- VARIANT TABLE EXTRACTION ---
variants = []
current_variant_lines = []
current_table = []
collecting_variant = False
collecting_table = False

# Process the table lines to extract variants and their tables
for line in table_lines:
    if is_variant_start(line):
        if current_variant_lines:
            variant_name = " ".join(current_variant_lines).replace("  ", " ")
            table = []
            last_line = ""
            for tbl_line in current_table:
                if (tbl_line.strip() and not is_radiation_level(tbl_line) and not re.match(r"^\w", tbl_line)):
                    last_line += " " + tbl_line.strip()
                else:
                    if last_line:
                        table.append(last_line.strip())
                    last_line = tbl_line.strip()
            if last_line:
                table.append(last_line.strip())
            parsed_table = parse_table_rows(table)
            variants.append({"variant": variant_name, "table": parsed_table})
        current_variant_lines = [line.strip()]
        current_table = []
        collecting_variant = True
        collecting_table = False
        continue
    if collecting_variant:
        if is_table_header(line):
            collecting_variant = False
            collecting_table = True
            current_table = [line.strip()]
        else:
            current_variant_lines.append(line.strip())
    elif collecting_table:
        if is_variant_start(line) or is_section_heading(line):
            collecting_table = False
        else:
            current_table.append(line.strip())

# Catch last variant : specific
if current_variant_lines:
    variant_name = " ".join(current_variant_lines).replace("  ", " ")
    table = []
    last_line = ""
    for tbl_line in current_table:
        if (tbl_line.strip() and not is_radiation_level(tbl_line) and not re.match(r"^\w", tbl_line)):
            last_line += " " + tbl_line.strip()
        else:
            if last_line:
                table.append(last_line.strip())
            last_line = tbl_line.strip()
    if last_line:
        table.append(last_line.strip())
    parsed_table = parse_table_rows(table)
    variants.append({"variant": variant_name, "table": parsed_table})

def clean_recommendations(lines):
    recs = []
    current = ""
    for line in lines:
        # Remove bullet
        line = line.lstrip("•").strip()
        # Find start of variant
        m = re.match(r'Variant \d:', line)
        if m:
            # Save the previous one if exists
            if current:
                recs.append(current.strip())
            current = line
        else:
            current += " " + line
    if current:
        recs.append(current.strip())
    # Remove summary header if present
    recs = [r for r in recs if not r.lower().startswith("summary of recommendations")]
    return recs

def clean_references(lines):
    refs = []
    current = ""
    for line in lines:
        # Detect start of new reference
        m = re.match(r'^\d+\.', line.strip())
        if m:
            if current:
                refs.append(current.strip())
            current = line.strip()
        else:
            # Continuation of previous reference
            current += " " + line.strip()
    if current:
        refs.append(current.strip())
    # Remove references header if present
    refs = [r for r in refs if not r.lower().startswith("references")]
    return refs

# --- Apply to your extracted slices to clean ---
summary_recommendations_lines = lines[start_summary_recommendations_index:start_summary_evidence_index] if (start_summary_recommendations_index is not None and start_summary_evidence_index is not None) else []
references_lines = lines[start_reference_index:] if start_reference_index is not None else []

summary_of_recommendations = clean_recommendations(summary_recommendations_lines)
references = clean_references(references_lines)

# --- BUILD FINAL JSON ---
result = {
    "category": current_category,
    "variants": variants,
    "Summary of Literature Review": " ".join(introduction_lines),
    "Summary of recommendations": summary_of_recommendations,
    #"summary_of_evidence": summary_evidence_lines,
    "References": references
}

# Note: this script does not ecompass conditions that are multi-line. You will need to export the json at this stage
# and process it further if needed. You can load it back for the next stages, which is to get the reasoning part.


def extract_variant_procedure_reasoning(lines):
    """
    Extract variant, procedure, and reasoning information from the reasoning lines. (after tables).
    These will be merged later.
    """
    out = []
    current_variant = None
    current_procedure = None
    buffer = []
    variant_name_buffer = []
    
    # here you will need to add all the procedure names found in the reasoning lines
    # this is not automated because the procedure names can vary in wording and can be multiline too.
    procedure_headers = {
    }


    for i, line in enumerate(lines):
        # Variant header
        if re.match(r"Variant \d:", line):
            # Save previous procedure if any
            if current_variant and current_procedure and buffer:
                out.append({
                    "variant": current_variant,
                    "procedure": current_procedure,
                    "reasoning": " ".join(buffer).strip()
                })
            current_variant = line.strip()
            # Capture any extra lines in variant name (next lines until a procedure header)
            variant_name_buffer = []
            current_procedure = None
            buffer = []
        # Extra lines belonging to the variant name (like the long description)
        elif current_variant and not current_procedure and not line in procedure_headers and not re.match(r"Variant \d:", line):
            variant_name_buffer.append(line.strip())
        # Procedure header
        elif current_variant and (line.strip() in procedure_headers or (current_procedure and line.strip() in procedure_headers)):
            # Save previous procedure if any
            if current_procedure and buffer:
                out.append({
                    "variant": current_variant + " " + " ".join(variant_name_buffer),
                    "procedure": current_procedure,
                    "reasoning": " ".join(buffer).strip()
                })
            current_procedure = line.strip()
            buffer = []
        # Reasoning text
        elif current_procedure:
            buffer.append(line.strip())
    # Final save
    if current_variant and current_procedure and buffer:
        out.append({
            "variant": current_variant + " " + " ".join(variant_name_buffer),
            "procedure": current_procedure,
            "reasoning": " ".join(buffer).strip()
        })
    return out

reasoning_lines = lines[end_review_index:start_summary_evidence_index] 
#print("Reasoning lines:", reasoning_lines)
reasoning_list = extract_variant_procedure_reasoning(reasoning_lines)
reasoning_list


# Now we want to merge the table context (includes rating e.g. May be Appropriate) with the reasoning text for the
# condition-variant-procedure triplet. This is a bit more tricky because sometimes in the tables the procedures
# are referred to as differently from the reasoning. (e.g. they do not include "with contrast" etc.)
# We use the following normalization map logic.

# Normalization map: maps normalized keys to exact canonical headers (tables) or list of headers (found in reasoning)
# Populate yourself. Example 
# {
#     "CT Head": "CT head with IV contrast",
#     "MR Brain": "MRI brain" #if unchanged
# }

PROCEDURE_NORMALIZATION_MAP: dict[str, str] = {}

def normalize_proc(proc_name):
    """Simple normalization of procedure names by lowercasing, removing punctuation, and extra spaces."""
    proc_name = proc_name.lower()
    proc_name = proc_name.replace("diagnostic", "").replace("imaging", "")
    proc_name = re.sub(r'[^a-z0-9 ]', ' ', proc_name)
    proc_name = re.sub(r'\s+', ' ', proc_name).strip()
    return proc_name

def extract_variant_num(variant_string):
    m = re.search(r"Variant (\d+)", variant_string, re.IGNORECASE)
    return int(m.group(1)) if m else None

# Build lookup: (variant_num, normalized procedure) -> reasoning
reasoning_lookup = {}
for r in reasoning_list:
    vnum = extract_variant_num(r["variant"])
    proc_norm = normalize_proc(r["procedure"])
    canonical = PROCEDURE_NORMALIZATION_MAP.get(proc_norm, r["procedure"])
    # Normalize canonical key too for consistent lookup
    canonical_norm = normalize_proc(canonical)
    reasoning_lookup[(vnum, canonical_norm)] = r["reasoning"]

# Assign reasoning to result's table entries after normalizing procedure names
for variant in result["variants"]:
    vnum = extract_variant_num(variant["variant"])
    for proc in variant["table"]:
        proc_norm = normalize_proc(proc["procedure"])
        proc["reasoning"] = reasoning_lookup.get((vnum, proc_norm), "")

# Now 'result' has populated reasoning fields.
result