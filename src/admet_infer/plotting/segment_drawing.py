import colorsys, math, pickle
from io import BytesIO

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D


BASE_COLORS = [
    (0.90, 0.35, 0.35), (0.35, 0.55, 0.95), (0.35, 0.75, 0.40),
    (0.90, 0.65, 0.20), (0.60, 0.40, 0.90), (0.35, 0.80, 0.85),
    (0.90, 0.40, 0.70), (0.50, 0.50, 0.50), (0.70, 0.70, 0.20),
    (0.20, 0.70, 0.20), (0.90, 0.55, 0.25), (0.45, 0.70, 0.95),
    (0.65, 0.85, 0.35), (0.80, 0.45, 0.80), (0.45, 0.85, 0.65),
]


def build_colors(n):
    colors, hue = list(BASE_COLORS[:n]), 0.07
    while len(colors) < n:
        hue = (hue + 0.61803398875) % 1
        colors.append(colorsys.hsv_to_rgb(hue, 0.65, 0.95))
    return colors


def load_bundle(path):
    with open(path, "rb") as handle: return pickle.load(handle)


def get_records(payload):
    for key in ["records", "segment_records", "segments"]:
        records = payload.get(key)
        if isinstance(records, (list, tuple)):
            return [x for x in records if isinstance(x, dict)]
    return []


def get_segment_order(payload, n_records=None):
    n_records = len(get_records(payload)) if n_records is None else int(n_records)
    topology = payload.get("topology") or {}
    order = payload.get("segment_order")
    if order is None and isinstance(topology, dict):
        order = topology.get("segment_order")
    if order is None:
        return list(range(n_records))

    output, used = [], set()
    for index in order:
        try:
            index = int(index)
        except Exception:
            continue
        if 0 <= index < n_records and index not in used:
            output.append(index)
            used.add(index)

    output.extend(i for i in range(n_records) if i not in used)
    return output


def payload_in_model_order(payload, n_segments=None, order=None):
    records = get_records(payload)
    order = get_segment_order(payload, len(records)) if order is None else list(order)
    if n_segments is not None:
        order = order[:max(0, int(n_segments))]

    output = dict(payload)
    output_records = []
    for model_pos, raw_pos in enumerate(order):
        if not 0 <= raw_pos < len(records): continue
        record = dict(records[raw_pos])
        record["source_segment_index"] = record.get("segment_index", raw_pos)
        record["source_segment_label"] = record.get("segment_label", f"S{raw_pos + 1}")
        record["source_component_index"] = record.get("component_index", raw_pos)
        record["source_order_position"] = record.get("order_position", raw_pos)
        record["source_segment_pos"] = raw_pos
        record["source_segment_number"] = raw_pos + 1
        record["segment_index"] = model_pos
        record["segment_label"] = f"S{model_pos + 1}"
        record["component_index"] = model_pos
        record["order_position"] = model_pos
        output_records.append(record)

    output["records"] = output_records
    output["segment_order"] = list(range(len(output["records"])))
    return output


def get_atom_ids(record):
    for key in ["atom_ids", "atoms", "atom_indices", "idxs"]:
        values = record.get(key)
        if values is None: continue
        output = []
        for value in values:
            try: output.append(int(value))
            except Exception: pass
        return sorted(set(output))
    return []


def has_drawable_records(payload):
    return isinstance(payload, dict) and any(get_atom_ids(x) for x in get_records(payload))


def get_drawable_cache(bundle):
    if not isinstance(bundle, dict): raise TypeError(f"Segment bundle must be dict, got {type(bundle)}")
    for key in ["segment_cache", "seq_tensor_cache", "cache"]:
        source = bundle.get(key)
        if isinstance(source, dict):
            cache = {str(k): v for k, v in source.items() if isinstance(v, dict)}
            if any(has_drawable_records(x) for x in cache.values()): return cache
    cache = {str(k): v for k, v in bundle.items() if isinstance(v, dict)}
    if any(has_drawable_records(x) for x in cache.values()): return cache
    raise RuntimeError("No atom-level segment records found; segment_cache must retain atom_ids.")


def get_smiles(payload, fallback=""):
    for key in ["smiles", "canonical_smiles", "SMILES", "SMILES_input"]:
        value = payload.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan": return str(value)
    return str(fallback)


def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is not None: return mol
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    if mol is None: return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        mol.UpdatePropertyCache(strict=False); Chem.GetSymmSSSR(mol)
        return mol
    except Exception:
        return None


def trim_white(image, padding=25):
    box = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox()
    if box is None: return image
    return image.crop((max(0, box[0] - padding), max(0, box[1] - padding),
                       min(image.width, box[2] + padding), min(image.height, box[3] + padding)))


def get_label_font(size):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception: return ImageFont.load_default()


def boxes_overlap(a, b, pad=3):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def structure_mask(image, margin=4, threshold=248):
    pixels = np.asarray(image.convert("RGB"))
    mask = np.any(pixels < threshold, axis=2).astype(np.uint8) * 255
    size = 2 * margin + 1
    return np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(size))) > 0


def box_has_structure(mask, box):
    left, top, right, bottom = map(int, box)
    crop = mask[max(0, top):min(mask.shape[0], bottom), max(0, left):min(mask.shape[1], right)]
    return crop.size > 0 and crop.any()


def rotate_vector(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def box_edge_point(anchor, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    dx, dy = anchor[0] - cx, anchor[1] - cy
    if abs(dx) < 1e-8 and abs(dy) < 1e-8: return cx, cy
    scale = min((box[2] - box[0]) / (2 * abs(dx)) if dx else np.inf,
                (box[3] - box[1]) / (2 * abs(dy)) if dy else np.inf)
    return cx + dx * scale, cy + dy * scale


def pointer_path(mask, anchor, end, occupied, clear_run=6):
    distance = max(abs(end[0] - anchor[0]), abs(end[1] - anchor[1]))
    count = max(2, int(distance) + 1)
    xs = np.rint(np.linspace(anchor[0], end[0], count)).astype(int)
    ys = np.rint(np.linspace(anchor[1], end[1], count)).astype(int)
    xs = np.clip(xs, 0, mask.shape[1] - 1); ys = np.clip(ys, 0, mask.shape[0] - 1)
    blocked = mask[ys, xs]

    start = None
    for i in range(2, max(2, len(blocked) - clear_run)):
        if not blocked[i:i + clear_run].any():
            start = i
            break
    if start is None or blocked[start:].any(): return None

    for box in occupied:
        inside = (xs[start:] >= box[0]) & (xs[start:] <= box[2]) & (ys[start:] >= box[1]) & (ys[start:] <= box[3])
        if inside.any(): return None
    return float(xs[start]), float(ys[start])


def place_segment_labels(image, labels, atom_points, font_size=18, distance=55):
    draw, font, occupied = ImageDraw.Draw(image), get_label_font(font_size), []
    mask = structure_mask(image, margin=4)
    mol_center = np.mean(atom_points, axis=0) if len(atom_points) else np.array([image.width / 2, image.height / 2])
    labels = sorted(labels, key=lambda x: math.atan2(x["anchor"][1] - mol_center[1], x["anchor"][0] - mol_center[0]))
    angles = [0] + [x for a in np.linspace(0.25, math.pi, 13) for x in (a, -a)]
    radii = [distance, distance + 20, distance + 45, distance + 75, distance + 110, distance + 150, distance + 200]

    for index, item in enumerate(labels):
        text, anchor = item["text"], np.asarray(item["anchor"], float)
        direction = anchor - mol_center
        if np.linalg.norm(direction) < 5:
            angle = 2 * math.pi * index / max(1, len(labels))
            direction = np.array([math.cos(angle), math.sin(angle)])
        direction /= max(np.linalg.norm(direction), 1e-8)

        bounds = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        box_w, box_h, selected = text_w + 10, text_h + 8, None

        for radius in radii:
            for delta in angles:
                dx, dy = rotate_vector(*direction, delta)
                center = anchor + radius * np.array([dx, dy])
                left = int(np.clip(center[0] - box_w / 2, 3, image.width - box_w - 3))
                top = int(np.clip(center[1] - box_h / 2, 3, image.height - box_h - 3))
                box = (left, top, left + box_w, top + box_h)
                if any(boxes_overlap(box, old, 6) for old in occupied) or box_has_structure(mask, box): continue
                end = box_edge_point(anchor, box)
                start = pointer_path(mask, anchor, end, occupied)
                if start is not None:
                    selected = box, start, end
                    break
            if selected is not None: break

        if selected is None:
            print(f"[DRAW][WARN] no clear label route for {text}; label omitted")
            continue

        box, start, end = selected
        occupied.append(box)
        color = tuple(int(255 * x) for x in item["color"])
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        draw.line([start, end], fill=color, width=2)
        draw.ellipse((start[0] - 2, start[1] - 2, start[0] + 2, start[1] + 2), fill=color)
        draw.rounded_rectangle(box, radius=4, fill="white", outline=color, width=2)
        draw.text((center[0] - text_w / 2, center[1] - text_h / 2 - 1), text, fill="black", font=font)
    return image


def ordered_ring(mol, ring):
    ring = set(ring)
    neighbors = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring] for i in ring}
    if any(len(x) != 2 for x in neighbors.values()): return None
    start = current = min(ring); previous, order = None, []
    while current not in order:
        order.append(current)
        choices = [x for x in neighbors[current] if x != previous]
        previous, current = current, choices[0]
    return order if len(order) == len(ring) and current == start else None


def largest_macrocycle(mol, min_size=9):
    for ring in sorted(Chem.GetSymmSSSR(mol), key=len, reverse=True):
        order = ordered_ring(mol, [int(x) for x in ring])
        if order and len(order) >= min_size: return order
    return None


def clean_macrocycle_ring(mol, ring):
    ring = set(ring)
    ring_sets = [set(int(x) for x in r) for r in Chem.GetSymmSSSR(mol)]
    fused = sum(1 for atom in ring if sum(atom in r for r in ring_sets) > 1)
    aromatic = sum(1 for atom in ring if mol.GetAtomWithIdx(atom).GetIsAromatic())
    return fused <= max(2, int(0.15 * len(ring))) and aromatic <= max(1, int(0.25 * len(ring)))


def copy_coords(target, source):
    conf = source.GetConformer()
    new_conf = Chem.Conformer(target.GetNumAtoms())
    for atom in range(target.GetNumAtoms()):
        new_conf.SetAtomPosition(atom, conf.GetAtomPosition(atom))
    target.RemoveAllConformers()
    target.AddConformer(new_conf)


def bond_segments(mol):
    conf = mol.GetConformer()
    output = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pa, pb = conf.GetAtomPosition(a), conf.GetAtomPosition(b)
        output.append((a, b, np.array([pa.x, pa.y]), np.array([pb.x, pb.y])))
    return output


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(a1, a2, b1, b2):
    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)


def layout_score(mol):
    if mol.GetNumConformers() == 0: return float("inf")
    segments = bond_segments(mol)
    if not segments: return 0.0

    crossings = 0
    for i, (a, b, p1, p2) in enumerate(segments):
        for c, d, q1, q2 in segments[i + 1:]:
            if len({a, b, c, d}) < 4: continue
            crossings += int(segments_cross(p1, p2, q1, q2))

    coords = np.array([
        [mol.GetConformer().GetAtomPosition(i).x, mol.GetConformer().GetAtomPosition(i).y]
        for i in range(mol.GetNumAtoms())
    ])
    width, height = np.ptp(coords[:, 0]), np.ptp(coords[:, 1])
    aspect = max(width / max(height, 1e-6), height / max(width, 1e-6))
    lengths = np.array([np.linalg.norm(p1 - p2) for _, _, p1, p2 in segments])
    long_bond = lengths.max() / max(np.median(lengths), 1e-6)
    return 100 * crossings + 4 * max(0, aspect - 4) + 6 * max(0, long_bond - 3)


def layout_coordgen(mol):
    rdDepictor.LoadDefaultRingSystemTemplates()
    with rdDepictor.UsingCoordGen(True):
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)


def non_ring_components(mol, ring_atoms):
    ring, seen, output = set(ring_atoms), set(), []
    for atom in range(mol.GetNumAtoms()):
        if atom in ring or atom in seen: continue
        stack, component = [atom], []
        while stack:
            current = stack.pop()
            if current in seen or current in ring: continue
            seen.add(current); component.append(current)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors())
        attachments = sorted({n.GetIdx() for i in component for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring})
        output.append((component, attachments))
    return output


def rotate_branches_outward(mol, ring_atoms):
    conf = mol.GetConformer()
    ring_xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in ring_atoms])
    center = ring_xy.mean(axis=0)

    for component, attachments in non_ring_components(mol, ring_atoms):
        if len(attachments) != 1 or not component: continue
        point = conf.GetAtomPosition(attachments[0]); anchor = np.array([point.x, point.y])
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in component])
        current, outward = coords.mean(axis=0) - anchor, anchor - center
        if np.linalg.norm(current) < 1e-6 or np.linalg.norm(outward) < 1e-6: continue
        angle = math.atan2(outward[1], outward[0]) - math.atan2(current[1], current[0])
        if abs(angle) < .35: continue
        c, s = math.cos(angle), math.sin(angle); rotation = np.array([[c, -s], [s, c]])
        for atom in component:
            point = conf.GetAtomPosition(atom)
            moved = anchor + rotation @ (np.array([point.x, point.y]) - anchor)
            conf.SetAtomPosition(atom, (float(moved[0]), float(moved[1]), 0.0))


def compute_good_2d_layout(mol):
    candidates = []

    def add_candidate(name, draw):
        candidate = Chem.Mol(mol)
        try:
            draw(candidate)
            rdDepictor.NormalizeDepiction(candidate)
            candidates.append((name, candidate))
        except Exception:
            pass

    if rdDepictor.IsCoordGenSupportAvailable():
        add_candidate("coordgen", layout_coordgen)

    add_candidate("rdkit", lambda m: (
        rdDepictor.LoadDefaultRingSystemTemplates(),
        rdDepictor.Compute2DCoords(
            m, canonOrient=True, clearConfs=True, nFlipsPerSample=4,
            nSample=300, sampleSeed=42, permuteDeg4Nodes=True,
            forceRDKit=True, useRingTemplates=True
        )
    ))

    ring = largest_macrocycle(mol)
    if ring and clean_macrocycle_ring(mol, ring):
        n = len(ring); radius = max(2.8, 1.5 / (2 * math.sin(math.pi / n)))
        coord_map = {atom: Point2D(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
                     for i, atom in enumerate(ring)}
        add_candidate("macrocycle", lambda m: (
            rdDepictor.Compute2DCoords(
                m, canonOrient=False, clearConfs=True, coordMap=coord_map,
                nFlipsPerSample=4, nSample=200, sampleSeed=42,
                permuteDeg4Nodes=True, forceRDKit=True, useRingTemplates=True
            ),
            rotate_branches_outward(m, ring)
        ))

    if not candidates:
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
        rdDepictor.NormalizeDepiction(mol)
        return

    scored = [(name, candidate, layout_score(candidate)) for name, candidate in candidates]
    _, best, best_score = min(scored, key=lambda x: x[2])
    macro = [x for x in scored if x[0] == "macrocycle"]
    if macro:
        _, macro_best, macro_score = min(macro, key=lambda x: x[2])
        if macro_score <= best_score + 30:
            best = macro_best
    copy_coords(mol, best)


def segment_anchor(points, mol_center):
    points = np.asarray(points, float)
    center = points.mean(axis=0)
    direction = center - mol_center
    if np.linalg.norm(direction) < 1e-6:
        return points[np.argmax(np.linalg.norm(points - mol_center, axis=1))]
    return points[np.argmax((points - mol_center) @ direction)]


def draw_segment_map(mol_id, payload, fallback_smiles="", width=1500, height=750):
    records, mol = get_records(payload), mol_from_smiles(get_smiles(payload, fallback_smiles))
    if not records or mol is None:
        image = Image.new("RGB", (width, height), "white")
        message = "No segment records" if not records else "RDKit failed to parse SMILES"
        ImageDraw.Draw(image).text((20, 20), f"{mol_id}\n{message}", fill="black")
        return image

    try: compute_good_2d_layout(mol)
    except Exception: rdDepictor.Compute2DCoords(mol)

    colors = build_colors(len(records))
    atom_segment = {atom: i for i, record in enumerate(records) for atom in get_atom_ids(record) if 0 <= atom < mol.GetNumAtoms()}
    atom_colors = {atom: colors[i] for atom, i in atom_segment.items()}
    highlight_bonds, bond_colors = [], {}

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in atom_segment and atom_segment.get(begin) == atom_segment.get(end):
            highlight_bonds.append(bond.GetIdx()); bond_colors[bond.GetIdx()] = colors[atom_segment[begin]]

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.padding, options.bondLineWidth, options.highlightRadius = .16, 1.8, .22
    options.fillHighlights, options.annotationFontScale = True, .7
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_segment), highlightAtomColors=atom_colors,
                        highlightBonds=highlight_bonds, highlightBondColors=bond_colors)

    atom_points = np.array([(drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y) for i in range(mol.GetNumAtoms())])
    mol_center, labels = atom_points.mean(axis=0), []

    for index, record in enumerate(records):
        atom_ids = [i for i in get_atom_ids(record) if 0 <= i < mol.GetNumAtoms()]
        if not atom_ids: continue
        points = np.array([(drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y) for i in atom_ids])
        labels.append({"text": f"S{index + 1}", "anchor": segment_anchor(points, mol_center), "color": colors[index]})

    drawer.FinishDrawing()
    image = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")
    font_size = 22 if len(records) <= 12 else 18 if len(records) <= 24 else 16
    return trim_white(place_segment_labels(image, labels, atom_points, font_size, 55), padding=35)
