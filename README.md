# Coffee roast evenness analyzer

Measures how evenly a batch of coffee is roasted from a photo, using computer
vision. Built for a hobby hand-turned drum roaster.

Each bean is segmented individually, its lightness (CIELAB **L\***) is
measured, and the spread across beans (coefficient of variation, CV) is
reported as the evenness score. Lower CV means a more even roast.

## Install

```
pip install -r requirements.txt
```

Use regular `opencv-python` (not `-headless`) if you want the interactive
region-selection window.

## Usage

Beans spread in a single layer on plain paper (most accurate):

```
python roast_evenness.py photo.jpg
python roast_evenness.py photo.jpg --light-beans   # beans lighter than background
```

Beans piled together with no background (e.g. in a cooling tray):

```
python roast_evenness.py photo.jpg --packed                       # drag a box
python roast_evenness.py photo.jpg --packed --roi 0.2 0.28 0.62 0.42
```

`--roi X Y W H` is a region as fractions of the image. Reuse the same values
across photos so results are comparable.

## Output

Written next to the photo:

- `<name>_annotated.png` - detected beans outlined with their L\* values (check this first)
- `<name>_beans.csv` - per-bean measurements
- `<name>_hist.png` - pixel-level and per-bean L\* distributions

Printed: bean count, mean and standard deviation of L\*, CV, and a pixel-level
comparison.

## Tips and limitations

- Compare **CV** between roasts, not absolute L\*, until lighting and exposure
  are fixed. Phone auto-exposure shifts absolute values. A gray card in frame
  would let you calibrate.
- Packed mode only sees the top layer, and misses heavily overlapped beans.
  Beans on paper is more accurate.
- Segmentation parameters (in `segment_packed`) are scale-dependent. Raise
  `seed_sigma` if beans split into pieces, lower it if neighbours merge.

## Roadmap

- RPM tracking of the drum from video
- Thermal camera temperature readout
- Lumped-parameter heat flow model and efficiency estimate
