#!/usr/bin/env python3
"""
Lithophane generator — enhanced prototype.

Supports:
- Linear or Beer's-law thickness mapping
- Flat, curved, or cylindrical shapes
- Controlled sampling resolution
- Optional mesh simplification
- Closed watertight STL export

Usage examples:
  python lithophane.py photo.jpg out.stl --width 100 --res 0.25
  python lithophane.py photo.jpg out.stl --map beer --mu 1.2 --shape curved --curve 120
  python lithophane.py photo.jpg out.stl --simplify 0.4
"""

import argparse
import numpy as np
from PIL import Image
import trimesh


def load_and_resize(path, width_mm, res_mm):
    img = Image.open(path).convert("L")
    n_x = max(2, int(round(width_mm / res_mm)) + 1)
    aspect = img.height / img.width
    n_y = max(2, int(round(n_x * aspect)))
    img = img.resize((n_x, n_y), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def height_map(gray, tmin, tmax, mode="linear", mu=1.0):
    """Map brightness (1=bright) to thickness."""
    if mode == "linear":
        return tmax - gray * (tmax - tmin)
    elif mode == "beer":
        # Approximate Beer's law: transmission ~ exp(-mu * t)
        # Want bright (gray=1) -> high transmission -> thin
        t_lo = np.exp(-mu * tmax)  # darkest transmission
        t_hi = np.exp(-mu * tmin)  # brightest transmission
        transmission = t_lo + gray * (t_hi - t_lo)
        # Avoid log(0)
        transmission = np.clip(transmission, 1e-6, 1.0)
        return -np.log(transmission) / mu
    else:
        raise ValueError(f"Unknown map mode: {mode}")


def build_mesh(z, res_mm, shape="flat", curve_deg=120.0):
    """
    Closed mesh from height field.
    shape: 'flat', 'curved', 'cylinder'
    """
    ny, nx = z.shape
    xs = np.arange(nx) * res_mm
    ys = (ny - 1 - np.arange(ny)) * res_mm
    X, Y = np.meshgrid(xs, ys)

    # Transform to 3D positions based on shape
    if shape == "flat":
        top_pos = np.stack([X, Y, z], axis=-1)
    else:
        # Curve in X direction
        width = xs[-1]
        if shape == "cylinder":
            radius = width / (2 * np.pi)
            theta = (X / width) * 2 * np.pi
        else:  # curved (arc)
            angle_rad = np.deg2rad(curve_deg)
            radius = width / angle_rad
            theta = (X / width - 0.5) * angle_rad

        # Place image on the curved surface; thickness is radial
        cx = radius * np.sin(theta)
        cy = Y
        # Base radius + thickness
        cz = radius * (1 - np.cos(theta)) if shape == "curved" else 0.0
        # Thickness outward
        radial = radius + z
        top_pos = np.stack([
            radial * np.sin(theta),
            cy,
            radial * np.cos(theta) - radius  # so it sits near z=0
        ], axis=-1)

    top = top_pos.reshape(-1, 3)

    def idx(r, c):
        return r * nx + c

    r, c = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    a = idx(r, c)
    b = idx(r, c + 1)
    d = idx(r + 1, c)
    e = idx(r + 1, c + 1)

    top_faces = np.vstack([
        np.stack([a, d, e], -1).reshape(-1, 3),
        np.stack([a, e, b], -1).reshape(-1, 3),
    ])

    # Bottom / inner surface
    if shape == "flat":
        bottom_z = np.zeros_like(z)
        bottom_pos = np.stack([X, Y, bottom_z], -1).reshape(-1, 3)
    else:
        # Inner radius
        if shape == "cylinder":
            radius = xs[-1] / (2 * np.pi)
            theta = (X / xs[-1]) * 2 * np.pi
        else:
            angle_rad = np.deg2rad(curve_deg)
            radius = xs[-1] / angle_rad
            theta = (X / xs[-1] - 0.5) * angle_rad
        radial_inner = radius  # or radius + small offset
        bottom_pos = np.stack([
            radial_inner * np.sin(theta),
            Y,
            radial_inner * np.cos(theta) - radius
        ], -1).reshape(-1, 3)

    # For simplicity in prototype, use a shared bottom construction for flat;
    # for curved we still build a closed shell using perimeter.
    # Full curved solid is more involved; this version keeps the top surface curved
    # and adds a simple flat-back approximation for printability in the prototype.

    # Fallback to original flat bottom method for robustness in this version
    # (curved top + approximate back). For production, a true solid of revolution is better.
    perimeter = []
    for x in range(nx - 1, 0, -1):
        perimeter.append(idx(0, x))
    for y in range(ny - 1):
        perimeter.append(idx(y, 0))
    for x in range(nx - 1):
        perimeter.append(idx(ny - 1, x))
    for y in range(ny - 1, 0, -1):
        perimeter.append(idx(y, nx - 1))
    per = np.array(perimeter, dtype=np.int32)

    n_top = len(top)
    bottom_verts = top[per].copy()
    bottom_verts[:, 2] = bottom_verts[:, 2].min() - 0.1  # simple offset back

    center = np.array([[bottom_verts[:, 0].mean(), bottom_verts[:, 1].mean(), bottom_verts[:, 2].mean()]])
    vertices = np.vstack([top, bottom_verts, center])

    n_per = len(per)
    center_idx = n_top + n_per
    bottom_faces = []
    for i in range(n_per):
        v0 = n_top + i
        v1 = n_top + ((i + 1) % n_per)
        bottom_faces.append([v0, v1, center_idx])
    bottom_faces = np.array(bottom_faces, dtype=np.int32)

    wall_faces = []
    for i in range(n_per):
        t0 = per[i]
        t1 = per[(i + 1) % n_per]
        b0 = n_top + i
        b1 = n_top + ((i + 1) % n_per)
        wall_faces.append([t0, b0, b1])
        wall_faces.append([t0, b1, t1])
    wall_faces = np.array(wall_faces, dtype=np.int32)

    faces = np.vstack([top_faces, bottom_faces, wall_faces])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Lithophane panel generator (enhanced prototype)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input image")
    parser.add_argument("output", help="Output STL")
    parser.add_argument("--width", type=float, default=100.0, help="Width in mm")
    parser.add_argument("--res", type=float, default=0.3, help="Sampling resolution (mm/pixel)")
    parser.add_argument("--tmin", type=float, default=0.8, help="Minimum thickness mm (bright)")
    parser.add_argument("--tmax", type=float, default=3.0, help="Maximum thickness mm (dark)")
    parser.add_argument("--map", choices=["linear", "beer"], default="linear",
                        help="Thickness mapping")
    parser.add_argument("--mu", type=float, default=1.0, help="Beer's law absorption coefficient")
    parser.add_argument("--shape", choices=["flat", "curved", "cylinder"], default="flat",
                        help="Panel shape")
    parser.add_argument("--curve", type=float, default=120.0, help="Curve angle in degrees (for curved)")
    parser.add_argument("--simplify", type=float, default=0.0,
                        help="Fraction of faces to keep after generation (0 = no simplify)")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    gray = load_and_resize(args.input, args.width, args.res)
    print(f"  Sampled: {gray.shape[1]} × {gray.shape[0]}")

    z = height_map(gray, args.tmin, args.tmax, args.map, args.mu)
    print(f"  Mapping: {args.map}  Thickness: {z.min():.2f} – {z.max():.2f} mm")

    print(f"Building {args.shape} mesh ...")
    mesh = build_mesh(z, args.res, args.shape, args.curve)
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print(f"  Watertight: {mesh.is_watertight}")

    if 0 < args.simplify < 1:
        target = max(200, int(len(mesh.faces) * args.simplify))
        print(f"Simplifying to ~{target:,} faces ...")
        mesh = mesh.simplify_quadric_decimation(face_count=target)
        print(f"  After: {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

    # Seat on Z=0 and center XY
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()
    mesh.vertices[:, :2] -= mesh.vertices[:, :2].mean(axis=0)

    mesh.export(args.output)
    print(f"Wrote {args.output}")
    print(f"  Final bounds: {mesh.bounds.round(1).tolist()}")


if __name__ == "__main__":
    main()
