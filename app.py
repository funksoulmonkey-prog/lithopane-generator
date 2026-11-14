#!/usr/bin/env python3
"""
Interactive Lithophane Generator (Streamlit)

Run with:
    streamlit run app.py
"""

import io
import numpy as np
from PIL import Image
import streamlit as st
import trimesh
import pyvista as pv

# Reuse core functions from the CLI version
def load_and_resize_from_pil(img: Image.Image, width_mm: float, height_mm: float, res_mm: float):
    img = img.convert("L")
    n_x = max(2, int(round(width_mm / res_mm)) + 1)
    n_y = max(2, int(round(height_mm / res_mm)) + 1)
    img = img.resize((n_x, n_y), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def height_map(gray, tmin, tmax, mode="linear", mu=1.0):
    if mode == "linear":
        return tmax - gray * (tmax - tmin)
    elif mode == "beer":
        t_lo = np.exp(-mu * tmax)
        t_hi = np.exp(-mu * tmin)
        transmission = t_lo + gray * (t_hi - t_lo)
        transmission = np.clip(transmission, 1e-6, 1.0)
        return -np.log(transmission) / mu
    raise ValueError(mode)


def build_mesh(z, res_mm, shape="flat", curve_deg=120.0):
    ny, nx = z.shape
    xs = np.arange(nx) * res_mm
    ys = (ny - 1 - np.arange(ny)) * res_mm
    X, Y = np.meshgrid(xs, ys)

    if shape == "flat":
        top_pos = np.stack([X, Y, z], axis=-1)
    else:
        width = xs[-1]
        if shape == "cylinder":
            radius = width / (2 * np.pi)
            theta = (X / width) * 2 * np.pi
        else:
            angle_rad = np.deg2rad(curve_deg)
            radius = width / angle_rad
            theta = (X / width - 0.5) * angle_rad
        radial = radius + z
        top_pos = np.stack([
            radial * np.sin(theta),
            Y,
            radial * np.cos(theta) - radius
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
    bottom_verts[:, 2] = bottom_verts[:, 2].min() - 0.1

    center = np.array([[bottom_verts[:, 0].mean(),
                        bottom_verts[:, 1].mean(),
                        bottom_verts[:, 2].mean()]])
    vertices = np.vstack([top, bottom_verts, center])

    n_per = len(per)
    center_idx = n_top + n_per
    bottom_faces = [[n_top + i, n_top + ((i + 1) % n_per), center_idx]
                    for i in range(n_per)]
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


# ── Streamlit UI ──────────────────────────────────────────────
st.set_page_config(page_title="Lithophane Generator", layout="wide")
st.title("Lithophane Panel Generator")
st.caption("Upload a photo → generate a 3D-printable lithophane STL")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload photo", type=["png", "jpg", "jpeg", "webp"])
    width = st.slider("Width (mm)", 40, 250, 100, 5)
    height = st.slider("Height (mm)", 40, 250, 75, 5)
    res = st.slider("Resolution (mm/pixel)", 0.15, 0.8, 0.3, 0.05)
    tmin = st.slider("Min thickness (mm)", 0.4, 2.0, 0.8, 0.1)
    tmax = st.slider("Max thickness (mm)", 1.5, 5.0, 3.0, 0.1)
    mapping = st.selectbox("Thickness map", ["linear", "beer"])
    mu = st.slider("Absorption μ (Beer's law)", 0.3, 3.0, 1.0, 0.1) if mapping == "beer" else 1.0
    shape = st.selectbox("Shape", ["flat", "curved", "cylinder"])
    curve = st.slider("Curve angle (°)", 30, 180, 120, 5) if shape == "curved" else 120
    simplify = st.slider("Simplify (keep fraction)", 0.1, 1.0, 0.6, 0.05)

with col2:
    if uploaded is None:
        st.info("Upload an image to generate a lithophane.")
    else:
        img = Image.open(uploaded)
        st.image(img, caption="Source", use_container_width=True)

        if st.button("Generate STL", type="primary"):
            with st.spinner("Building mesh…"):
                gray = load_and_resize_from_pil(img, width, height, res)
                z = height_map(gray, tmin, tmax, mapping, mu)
                mesh = build_mesh(z, res, shape, curve)

                if simplify < 1.0:
                    target = max(200, int(len(mesh.faces) * simplify))
                    mesh = mesh.simplify_quadric_decimation(face_count=target)

                mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()
                mesh.vertices[:, :2] -= mesh.vertices[:, :2].mean(axis=0)

                # Export to bytes
                buf = io.BytesIO()
                mesh.export(buf, file_type="stl")
                stl_bytes = buf.getvalue()

            st.success(f"Done — {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")
            st.download_button(
                label="Download STL",
                data=stl_bytes,
                file_name="lithophane.stl",
                mime="model/stl",
            )
            st.caption(f"Thickness range: {z.min():.2f} – {z.max():.2f} mm  |  "
                       f"Bounds: {mesh.bounds.round(1).tolist()}")

            # 3D preview
            st.subheader("3D Preview")
            pv_mesh = pv.wrap(mesh)
            plotter = pv.Plotter(window_size=[600, 400], off_screen=True)
            plotter.add_mesh(pv_mesh, color="white", show_edges=False, lighting=True)
            plotter.view_isometric()
            plotter.set_background("black")
            st.pyvista_chart(plotter)

st.markdown("---")
st.markdown("**Print tips:** vertical orientation, white/natural PLA, 0.1–0.12 mm layers, solid walls.")
