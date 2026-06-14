import os

def extract_block(lines, start_marker, end_marker):
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if start_marker in line and start_idx == -1:
            start_idx = i
        elif end_marker in line and start_idx != -1 and end_idx == -1:
            end_idx = i
            break
    if start_idx != -1 and end_idx != -1:
        return lines[start_idx:end_idx]
    if start_idx != -1:
        return lines[start_idx:]
    return []

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

out = []

# 1. Setup and imports (everything up to Sidebar)
out.extend(extract_block(lines, '"""', '# SIDEBAR'))

# 2. Add Sidebar opening
out.append("# ─────────────────────────────────────────────────────────────────────────────\n")
out.append("# SIDEBAR\n")
out.append("# ─────────────────────────────────────────────────────────────────────────────\n")
out.append("with st.sidebar:\n")
out.append("    st.markdown(\"# Veil Finder Mobile\")\n")
out.append("    st.caption(\"Read-only field viewer\")\n")

# 3. Hunt selector
out.extend(extract_block(lines, '# ── Hunt Selector ──', 'with st.expander("➕ Create New Hunt", expanded=False):'))
out.append("    st.markdown(\"---\")\n")

# 4. We want to skip satellite data upload, threshold setting, fog stats, grid settings, and plant addition.
# We do want the Combined Score Weights
out.extend(extract_block(lines, 'with st.expander("⚖️ Combined Score Weights"', 'with st.expander("🖼️ Map Image Overlays"'))

# 5. Overlays (Read-only version)
out.append('    with st.expander("🖼️ Map Image Overlays", expanded=False):\n')
out.append('        if "map_overlays" not in st.session_state:\n')
out.append('            st.session_state.map_overlays = _saved_settings.get("map_overlays", [])\n')
out.append('        for idx, overlay in enumerate(st.session_state.map_overlays):\n')
out.append('            st.markdown(f"**{overlay[\'name\']}**")\n')
out.append('            vis = st.checkbox("Visible", value=overlay.get("visible", True), key=f"ov_vis_{idx}")\n')
out.append('            if vis != overlay.get("visible", True):\n')
out.append('                overlay["visible"] = vis\n')
out.append('                st.session_state.settings_cache = load_settings()\n')
out.append('                st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays\n')
out.append('                save_settings(st.session_state.settings_cache)\n')
out.append('                st.rerun()\n')
out.append('            if vis:\n')
out.append('                n_op = st.slider("Opacity", 0.0, 1.0, value=float(overlay["opacity"]), step=0.05, key=f"ov_op_{idx}")\n')
out.append('                if n_op != overlay["opacity"]:\n')
out.append('                    overlay["opacity"] = n_op\n')
out.append('                    st.session_state.settings_cache = load_settings()\n')
out.append('                    st.session_state.settings_cache["map_overlays"] = st.session_state.map_overlays\n')
out.append('                    save_settings(st.session_state.settings_cache)\n')
out.append('                    st.rerun()\n')
out.append('            st.markdown("---")\n')

# 6. Map View
out.append("\n# ─── Map View ──────────────────────────────────────────────────────────────\n")
out.append("st.header(\"Location Scoring Map\")\n\n")
out.append("fog_df = get_fog_df()\n")
out.append("if fog_df is None:\n")
out.append("    st.warning(\"No satellite data found for this hunt.\")\n")
out.append("elif not st.session_state.observations:\n")
out.append("    st.info(\"No observations loaded for this hunt.\")\n")
out.append("else:\n")
out.append("    scores = get_scores()\n")
out.append("    if scores is not None and not scores.empty:\n")
out.append("        scores = recompute_combined(scores)\n\n")

# Extract everything from "# Summary metrics" until "# ─── TAB 4:"
map_logic = extract_block(lines, '# Summary metrics', '# ─── TAB 4: Diagnostics')

# Unindent by 4 spaces because they were inside `with tab3:` and `else:` (total 8 spaces)
for line in map_logic:
    if line.startswith("        "):
        out.append(line[4:])
    elif line.startswith("    "):
        out.append(line)
    else:
        out.append(line)

with open("field_app.py", "w", encoding="utf-8") as f:
    f.writelines(out)
