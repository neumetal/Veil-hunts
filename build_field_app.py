import os

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("field_app.py", "w", encoding="utf-8") as f:
    for line in lines:
        if line.startswith("# ─────────────────────────────────────────────────────────────────────────────"):
            # Check if next line is TABS
            idx = lines.index(line)
            if lines[idx+1].startswith("# TABS"):
                break
        f.write(line)

    f.write("\n# ─── Map View ──────────────────────────────────────────────────────────────\n")
    f.write("st.header(\"Location Scoring Map\")\n\n")
    
    f.write("fog_df = get_fog_df()\n")
    f.write("if fog_df is None:\n")
    f.write("    st.warning(\"Select a satellite data CSV in the sidebar.\")\n")
    f.write("elif not st.session_state.observations:\n")
    f.write("    st.info(\"No observations loaded for this hunt.\")\n")
    f.write("else:\n")
    f.write("    scores = get_scores()\n")
    f.write("    if scores is not None and not scores.empty:\n")
    f.write("        scores = recompute_combined(scores)\n\n")
    
    map_start_idx = 0
    for i in range(len(lines)):
        if lines[i].strip() == "# Summary metrics":
            map_start_idx = i
            break
            
    for i in range(map_start_idx, len(lines)):
        line = lines[i]
        
        if line.startswith("with tab4:"):
            break
            
        if line.startswith("            "):
            f.write(line[4:])
        elif line.startswith("        "):
            f.write(line)
        else:
            if line.strip() != "":
                f.write(line)
