# Lithophane Panel Generator

Convert photos into 3D-printable lithophane panels.

## Interactive App (recommended)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL (usually http://localhost:8501), upload a photo, tweak settings, and download the STL.

## CLI
```bash
python lithophane.py input.jpg output.stl [options]
```

### Options
- `--width` mm
- `--res` sampling mm/pixel
- `--tmin` / `--tmax`
- `--map linear|beer`
- `--mu` (for beer)
- `--shape flat|curved|cylinder`
- `--curve` degrees
- `--simplify` fraction to keep

## Print tips
Vertical orientation, white/natural PLA, 0.1–0.12 mm layers, solid walls / high perimeter count.
