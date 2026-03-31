"""
visualize_meshes.py: LV template mesh inspection + GIF export tool

Interactive mode (default):
  click       — select / deselect a mesh (turns red)
  d           — discard all selected meshes on current page
  n           — next page
  p           — previous page
  right/left  — next / previous cardiac phase
  q           — quit and save log

GIF mode:  python visualize_meshes.py --gifs
  Saves per-variant GIFs and a combined grid GIF to gifs/

Outputs:
  inspection_log.txt — list of discarded mesh variants
  gifs/              — animated GIFs of cardiac motion
"""

import glob
import os
import sys

import numpy as np
import pyvista as pv
import vtk

TEMPLATE_DIR = "data/template_LV/LV"
OUTPUT_LOG = "inspection_log.txt"
N_PHASES = 10
COLS = 3
ROWS = 2
PER_PAGE = COLS * ROWS
SPACING = 2.5

COLOR_DEFAULT   = "lightblue"
COLOR_SELECTED  = "tomato"
COLOR_DISCARDED = "gray"


def normalize_and_place_points(mesh, col, row):
    pts = mesh.points.copy()
    pts -= pts.mean(axis=0)
    span = max(
        mesh.bounds[1] - mesh.bounds[0],
        mesh.bounds[3] - mesh.bounds[2],
        mesh.bounds[5] - mesh.bounds[4],
    )
    if span > 0:
        pts /= span
    pts += np.array([col * SPACING, -row * SPACING, 0.0])
    return pts


def set_actor_color(actor, color_name):
    r, g, b = pv.Color(color_name).float_rgb
    actor.GetProperty().SetColor(r, g, b)
    actor.GetProperty().SetOpacity(1.0)


def run():
    variant_dirs = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "mesh_*")))
    if not variant_dirs:
        print(f"No mesh variants found in {TEMPLATE_DIR}")
        return
    print(f"Found {len(variant_dirs)} variants")

    # Store file paths only — load meshes lazily per page to avoid OOM
    # all_variants[i] = (name, [path_phase0, ..., path_phase9])
    all_variants = []
    for vdir in variant_dirs:
        phase_paths = []
        for p in range(N_PHASES):
            fn = os.path.join(vdir, f"phase{p}.vtp")
            phase_paths.append(fn if os.path.exists(fn) else None)
        all_variants.append((os.path.basename(vdir), phase_paths))

    def load_batch_phases(batch_paths):
        """Load meshes for the current page only."""
        loaded = []
        for name, phase_paths in batch_paths:
            phases = []
            for fn in phase_paths:
                if fn is None:
                    phases.append(None)
                    continue
                try:
                    phases.append(pv.read(fn))
                except Exception as e:
                    print(f"Could not read {fn}: {e}")
                    phases.append(None)
            loaded.append((name, phases))
        return loaded

    discarded = set()
    n_pages = (len(all_variants) + PER_PAGE - 1) // PER_PAGE
    page = [0]
    phase = [0]

    while True:
        start = page[0] * PER_PAGE
        batch = load_batch_phases(all_variants[start : start + PER_PAGE])
        selected = set()

        pl = pv.Plotter(
            title=f"Page {page[0]+1}/{n_pages}  |  Phase {phase[0]}  |  click=select  d=discard  n=next  p=prev  ←/→=phase  q=quit"
        )
        pl.set_background("darkgray")

        # actors[i] = (actor, placed_mesh, variant_name)
        actors = []
        for i, (name, phases) in enumerate(batch):
            row, col = divmod(i, COLS)
            mesh = phases[phase[0]]
            if mesh is None:
                actors.append((None, None, name))
                continue

            placed = mesh.copy()
            placed.points = normalize_and_place_points(mesh, col, row)
            color = COLOR_DISCARDED if name in discarded else COLOR_DEFAULT
            actor = pl.add_mesh(placed, color=color, show_edges=False, pickable=True)
            actors.append((actor, placed, name))

            label_pos = np.array([col * SPACING, -row * SPACING + 0.62, 0.0])
            pl.add_point_labels(
                [label_pos], [name],
                font_size=9, text_color="white",
                always_visible=True, show_points=False,
            )

        phase_actor = pl.add_text(
            f"Phase: {phase[0]}", position="upper_right", font_size=12, color="yellow"
        )

        # click to select / deselect meshes
        prop_picker = vtk.vtkPropPicker()

        def on_left_click(interactor, _event):
            x, y = interactor.GetEventPosition()
            prop_picker.Pick(x, y, 0, pl.renderer)
            if prop_picker.GetActor() is None:
                return
            pos = prop_picker.GetPickPosition()
            col_hit = int(round(pos[0] / SPACING))
            row_hit = int(round(-pos[1] / SPACING))
            idx = row_hit * COLS + col_hit
            if not (0 <= col_hit < COLS and 0 <= row_hit < ROWS and 0 <= idx < len(actors)):
                return
            actor, _, name = actors[idx]
            if actor is None:
                return
            if idx in selected:
                selected.discard(idx)
                set_actor_color(actor, COLOR_DISCARDED if name in discarded else COLOR_DEFAULT)
                print(f"  Deselected: {name}")
            else:
                selected.add(idx)
                set_actor_color(actor, COLOR_SELECTED)
                print(f"  Selected:   {name}")
            pl.render()

        pl.iren.interactor.AddObserver("LeftButtonPressEvent", on_left_click)

        # discard selected
        def discard_selected():
            if not selected:
                print("  Nothing selected.")
                return
            for idx in list(selected):
                actor, _, name = actors[idx]
                discarded.add(name)
                if actor is not None:
                    set_actor_color(actor, COLOR_DISCARDED)
            selected.clear()
            pl.render()
            print(f"  Discarded so far: {len(discarded)}")

        # phase cycling
        def change_phase(delta):
            phase[0] = (phase[0] + delta) % N_PHASES
            for i, (_, placed, _name) in enumerate(actors):
                if placed is None:
                    continue
                row, col = divmod(i, COLS)
                _, phases = batch[i]
                new_mesh = phases[phase[0]]
                if new_mesh is None:
                    continue
                placed.points = normalize_and_place_points(new_mesh, col, row)
                placed.GetPoints().Modified()
                placed.Modified()
            phase_actor.SetText(3, f"Phase: {phase[0]}")
            pl.render()
            print(f"  Phase: {phase[0]}")

        # auto-play timer
        playing = [True]

        def on_timer(_obj, _event):
            if playing[0]:
                change_phase(+1)

        pl.iren.interactor.AddObserver("TimerEvent", on_timer)
        timer_id = [pl.iren.interactor.CreateRepeatingTimer(150)]  # ms per frame

        def toggle_play():
            playing[0] = not playing[0]
            print(f"  {'Playing' if playing[0] else 'Paused'}")

        def stop_timer():
            playing[0] = False
            if timer_id[0] is not None:
                pl.iren.interactor.DestroyTimer(timer_id[0])
                timer_id[0] = None

        # page navigation
        navigated = [False]

        def next_page():
            if page[0] < n_pages - 1:
                page[0] += 1
            navigated[0] = True
            stop_timer()
            pl.iren.terminate_app()

        def prev_page():
            if page[0] > 0:
                page[0] -= 1
            navigated[0] = True
            stop_timer()
            pl.iren.terminate_app()

        pl.add_key_event("d", discard_selected)
        pl.add_key_event("n", next_page)
        pl.add_key_event("p", prev_page)
        pl.add_key_event("Right", lambda: change_phase(+1))
        pl.add_key_event("Left",  lambda: change_phase(-1))
        pl.add_key_event("space", toggle_play)
        pl.add_text(
            "click=select  d=discard  n=next  p=prev  ←/→=phase  space=pause  q=quit",
            position="lower_left", font_size=9, color="white",
        )

        pl.show()
        pl.close()

        if not navigated[0]:
            break

    with open(OUTPUT_LOG, "w") as f:
        f.write("Discarded meshes:\n")
        for name in sorted(discarded):
            f.write(f"{name}\n")

    print(f"\nDone. {len(discarded)} discarded. Log saved to {OUTPUT_LOG}")


def make_gifs(output_dir="gifs"):
    """Render all phases off-screen and save per-variant + combined GIFs."""
    os.makedirs(output_dir, exist_ok=True)

    variant_dirs = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "mesh_*")))
    if not variant_dirs:
        print(f"No mesh variants found in {TEMPLATE_DIR}")
        return
    print(f"Found {len(variant_dirs)} variants — generating GIFs in '{output_dir}/'")

    # Load all variants and phases
    all_variants = []
    for vdir in variant_dirs:
        phases = []
        for p in range(N_PHASES):
            fn = os.path.join(vdir, f"phase{p}.vtp")
            try:
                phases.append(pv.read(fn))
            except Exception as e:
                print(f"  Could not read {fn}: {e}")
                phases.append(None)
        all_variants.append((os.path.basename(vdir), phases))

    # Ping-pong frame sequence: 0→9→0 for a smooth loop
    frame_seq = list(range(N_PHASES)) + list(range(N_PHASES - 2, 0, -1))

    # Individual GIF per variant
    for name, phases in all_variants:
        ref = phases[0]
        if ref is None:
            continue
        # Compute normalization from phase 0 (kept constant across all phases)
        ref_center = ref.points.mean(axis=0)
        ref_span = max(
            ref.bounds[1] - ref.bounds[0],
            ref.bounds[3] - ref.bounds[2],
            ref.bounds[5] - ref.bounds[4],
        )

        pl = pv.Plotter(off_screen=True, window_size=[500, 500])
        pl.set_background("darkgray")

        placed = ref.copy()
        placed.points = (ref.points - ref_center) / ref_span
        pl.add_mesh(placed, color=COLOR_DEFAULT, show_edges=False)
        pl.add_text(name, position="upper_left", font_size=10, color="white")
        phase_label = pl.add_text("Phase: 0", position="upper_right",
                                  font_size=10, color="yellow")
        pl.view_isometric()
        pl.camera.zoom(1.4)

        out_path = os.path.join(output_dir, f"{name}.gif")
        pl.open_gif(out_path)
        for p in frame_seq:
            m = phases[p]
            if m is None:
                continue
            placed.points = (m.points - ref_center) / ref_span
            placed.GetPoints().Modified()
            placed.Modified()
            phase_label.SetText(2, f"Phase: {p}")   # corner 2 = upper-right
            pl.write_frame()
        pl.close()
        print(f"  Saved {out_path}")

    # Combined grid GIF (all variants together)
    out_path = os.path.join(output_dir, "all_variants.gif")
    pl = pv.Plotter(off_screen=True, window_size=[COLS * 350, ROWS * 350])
    pl.set_background("darkgray")

    # Compute per-variant normalization params once from phase 0
    norm_params = []
    for name, phases in all_variants:
        ref = phases[0]
        if ref is None:
            norm_params.append(None)
            continue
        center = ref.points.mean(axis=0)
        span = max(
            ref.bounds[1] - ref.bounds[0],
            ref.bounds[3] - ref.bounds[2],
            ref.bounds[5] - ref.bounds[4],
        )
        norm_params.append((center, span))

    placed_meshes = []
    for i, (name, phases) in enumerate(all_variants):
        row, col = divmod(i, COLS)
        ref = phases[0]
        if ref is None or norm_params[i] is None:
            placed_meshes.append(None)
            continue
        center, span = norm_params[i]
        placed = ref.copy()
        pts = (ref.points - center) / span
        pts += np.array([col * SPACING, -row * SPACING, 0.0])
        placed.points = pts
        pl.add_mesh(placed, color=COLOR_DEFAULT, show_edges=False)
        placed_meshes.append(placed)

        label_pos = np.array([col * SPACING, -row * SPACING + 0.62, 0.0])
        pl.add_point_labels([label_pos], [name], font_size=9,
                            text_color="white", always_visible=True, show_points=False)

    phase_label = pl.add_text("Phase: 0", position="upper_right",
                              font_size=12, color="yellow")
    pl.view_xy()
    pl.camera.zoom(0.9)

    pl.open_gif(out_path)
    for p in frame_seq:
        for i, (name, phases) in enumerate(all_variants):
            pm = placed_meshes[i]
            if pm is None or norm_params[i] is None:
                continue
            m = phases[p]
            if m is None:
                continue
            center, span = norm_params[i]
            row, col = divmod(i, COLS)
            pts = (m.points - center) / span
            pts += np.array([col * SPACING, -row * SPACING, 0.0])
            pm.points = pts
            pm.GetPoints().Modified()
            pm.Modified()
        phase_label.SetText(3, f"Phase: {p}")
        pl.write_frame()
    pl.close()
    print(f"  Saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    if "--gifs" in sys.argv or "-g" in sys.argv:
        make_gifs()
    else:
        run()
