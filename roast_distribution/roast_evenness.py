#!/usr/bin/env python3
"""
Roast evenness analyzer.

Usage:
    python roast_evenness.py photo.jpg                  # beans on plain paper
    python roast_evenness.py photo.jpg --light-beans    # beans lighter than background
    python roast_evenness.py photo.jpg --packed         # piled beans, drag a box
    python roast_evenness.py photo.jpg --packed --roi 0.2 0.28 0.62 0.42

Default mode assumes roasted beans spread in a single layer on a plain
background that contrasts with them. --packed handles beans piled together
with no background; only the region of interest is analyzed.

Outputs (next to the photo):
    <name>_annotated.png   beans outlined and labeled with their lightness
    <name>_beans.csv       per-bean measurements
    <name>_hist.png        histogram of per-bean lightness
and prints summary stats.

Requires: opencv-python, numpy, matplotlib, scipy, scikit-image (last two: --packed only)
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def _odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def segment_packed(img_bgr, roi_mask, bean_px=28):
    """Segment beans that are piled together with no background.

    Idea: treat the whole region of interest as "bean", find ONE seed per
    bean with a blob detector sized to a bean, then flood outward on the
    inverted brightness map so the dark gaps between beans act as ridges.

    bean_px: approximate bean WIDTH in pixels at the working resolution
    (1600 px on the long side). Everything below scales from it.
    """
    h, w = img_bgr.shape[:2]
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    K = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    # 1. Seeds: local maxima of a difference-of-Gaussians (bean-sized blobs)
    s1 = 0.21 * bean_px
    dog = cv2.GaussianBlur(L, (0, 0), s1) - cv2.GaussianBlur(L, (0, 0), 2 * s1)
    win = _odd(0.68 * bean_px)
    peaks = (dog == cv2.dilate(dog, np.ones((win, win), np.uint8))) & (dog > 1.0)
    peaks &= roi_mask > 0
    seeds = cv2.dilate(peaks.astype(np.uint8) * 255, K(7))
    _, markers = cv2.connectedComponents(seeds)
    markers = markers + 1
    markers[(markers == 1) & (roi_mask > 0)] = 0   # unlabeled -> to be flooded
    markers[roi_mask == 0] = 1                     # outside ROI = background

    # 2. Flood on inverted brightness: bean bodies = basins, gaps = ridges
    elev = cv2.GaussianBlur(255.0 - L, (0, 0), 2)
    elev = cv2.normalize(elev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    markers = cv2.watershed(cv2.cvtColor(elev, cv2.COLOR_GRAY2BGR),
                            markers.astype(np.int32))
    labels = np.where(markers > 1, markers - 1, 0).astype(np.int32)
    labels[roi_mask == 0] = 0

    # 3. Clean each region: trim thin tendrils that leak along gaps and keep
    #    the largest piece, so outlines look like beans.
    ksz = _odd(0.32 * bean_px)
    out = np.zeros_like(labels)
    for i in np.unique(labels):
        if i == 0:
            continue
        m = cv2.morphologyEx((labels == i).astype(np.uint8), cv2.MORPH_OPEN, K(ksz))
        n, cc, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            out[cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])] = i
    return out


def segment_packed(img_bgr, roi_mask=None, win=51, seed_sigma=5.0, min_dist=10):
    """Segment beans that are piled together with no background.

    1. Two masks from a local (adaptive) threshold on L: a generous one that
       includes dark gaps/shadows, and a tight one that is only the bright
       bean bodies.
    2. Seeds = peaks of a heavily blurred L image (roughly one per bean, and
       the blur merges a bean's crease and body into a single bump).
    3. Watershed floods a lightly blurred -L landscape from those seeds, so
       region boundaries fall along the dark gaps between beans.
    4. Each region is trimmed to the tight mask, holes are filled (this puts
       the pale crease back in), thin tendrils are removed, and 2 px are
       eroded off so shadowed edges don't drag the brightness down.
    """
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    Lb = cv2.GaussianBlur(L, (0, 0), 1.5)
    L8 = np.clip(Lb, 0, 255).astype(np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def local_mask(C):
        m = cv2.adaptiveThreshold(L8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, win, C) > 0
        if roi_mask is not None:
            m &= roi_mask > 0
        return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k5,
                                iterations=2) > 0

    generous, tight = local_mask(8), local_mask(2)

    Ls = cv2.GaussianBlur(L, (0, 0), seed_sigma)
    peaks = peak_local_max(Ls, min_distance=min_dist,
                           labels=tight.astype(int), exclude_border=False)
    markers = np.zeros(L.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    regions = watershed(-Lb, markers, mask=generous)

    labels = np.zeros(L.shape, np.int32)
    for i in range(1, regions.max() + 1):
        r = (regions == i) & tight
        if r.sum() < 80:
            continue
        r = ndi.binary_fill_holes(r)
        r = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_OPEN, k11)
        n, cc, st, _ = cv2.connectedComponentsWithStats(r)
        if n < 2:
            continue
        r = (cc == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        r = cv2.erode(r, k_er)
        labels[r > 0] = i
    return labels


def segment_beans(img_bgr, beans_darker=True, roi_mask=None, packed=False):
    """Return a label image: 0 = background, 1..N = individual beans.

    packed=True is for beans piled together with no background (e.g. in a
    tray); see segment_packed. Otherwise beans are assumed to sit on a
    contrasting background.
    """
    if packed:
        return segment_packed(img_bgr, roi_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    flag = cv2.THRESH_BINARY_INV if beans_darker else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    # Watershed to split touching beans
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, seeds = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    seeds = seeds.astype(np.uint8)
    _, markers = cv2.connectedComponents(seeds)
    markers = markers + 1
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, seeds)
    markers[unknown == 255] = 0
    markers = cv2.watershed(img_bgr, markers.astype(np.int32))

    labels = np.zeros_like(markers)
    ok = (markers > 1) & (mask > 0)
    labels[ok] = markers[ok] - 1
    return labels


def measure_beans(img_bgr, labels, min_area_frac=0.4, max_area_frac=2.2,
                  min_solidity=0.80):
    """Per-bean mean L* (0-100), excluding specular highlights."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0  # OpenCV scales L to 0-255

    ids = [i for i in np.unique(labels) if i != 0]
    areas = np.array([(labels == i).sum() for i in ids])
    if len(areas) == 0:
        return []
    median_area = np.median(areas)

    results = []
    for i, area in zip(ids, areas):
        # drop tiny fragments (noise) and huge blobs (merged beans)
        if area < min_area_frac * median_area or area > max_area_frac * median_area:
            continue
        m = labels == i
        # drop ragged / merged shapes: area relative to convex hull area
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
        if hull_area > 0 and area / hull_area < min_solidity:
            continue
        vals = L[m]
        # Mask specular glare: brightest pixels within the bean
        cutoff = np.percentile(vals, 90)
        vals = vals[vals <= cutoff]
        ys, xs = np.nonzero(m)
        results.append(
            dict(id=int(i), L=float(vals.mean()), area=int(area),
                 cx=float(xs.mean()), cy=float(ys.mean()))
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--light-beans", action="store_true",
                    help="beans are lighter than the background")
    ap.add_argument("--packed", action="store_true",
                    help="beans are piled together with no background "
                         "(implies choosing a region of interest)")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="region of interest as fractions of the image, "
                         "e.g. --roi 0.2 0.3 0.6 0.4. If omitted with "
                         "--packed, a window opens so you can drag a box.")
    args = ap.parse_args()

    path = Path(args.image)
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read {path}")

    # Downscale big phone photos for speed / consistent kernel sizes
    scale = 1600 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    roi_mask = None
    if args.roi or args.packed:
        h, w = img.shape[:2]
        if args.roi:
            x, y, rw, rh = args.roi
            x, y, rw, rh = int(x * w), int(y * h), int(rw * w), int(rh * h)
        else:
            print("Drag a box over beans only (no rim/table), then press ENTER.")
            x, y, rw, rh = cv2.selectROI("Select beans region", img, False)
            cv2.destroyAllWindows()
            if rw == 0 or rh == 0:
                raise SystemExit("No region selected.")
        roi_mask = np.zeros(img.shape[:2], np.uint8)
        roi_mask[y:y + rh, x:x + rw] = 255

    labels = segment_beans(img, beans_darker=not args.light_beans,
                           roi_mask=roi_mask, packed=args.packed)
    beans = measure_beans(img, labels)
    if len(beans) < 5:
        raise SystemExit(f"Only found {len(beans)} beans - check contrast/background.")

    L = np.array([b["L"] for b in beans])
    mean, std = L.mean(), L.std(ddof=1)
    print(f"Beans measured : {len(L)}")
    print(f"Mean L*        : {mean:.1f}  (lower = darker roast)")
    print(f"Std dev L*     : {std:.2f}")
    print(f"CV             : {100 * std / mean:.1f}%  (lower = more even)")
    print(f"Range (P5-P95) : {np.percentile(L, 5):.1f} - {np.percentile(L, 95):.1f}")

    # Pixel-level distribution: every pixel inside the kept beans
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * 100 / 255
    keep_mask = np.isin(labels, [b["id"] for b in beans])
    pix = lab[keep_mask]
    pmean, pstd = pix.mean(), pix.std(ddof=1)
    print()
    print(f"Pixels measured: {len(pix):,}")
    print(f"Pixel mean L*  : {pmean:.1f}")
    print(f"Pixel std L*   : {pstd:.2f}")
    print(f"Pixel CV       : {100 * pstd / pmean:.1f}%  "
          f"(includes within-bean shading and glare)")

    # Annotated image
    out = img.copy()
    for b in beans:
        m = (labels == b["id"]).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
        cv2.putText(out, f"{b['L']:.0f}", (int(b["cx"]) - 12, int(b["cy"]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path.with_name(path.stem + "_annotated.png")), out)

    with open(path.with_name(path.stem + "_beans.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "L", "area", "cx", "cy"])
        w.writeheader()
        w.writerows(beans)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Zoom to the bulk of the data so a few glare pixels don't stretch the axis
    lo = min(np.percentile(pix, 1), L.min()) - 2
    hi = max(np.percentile(pix, 97), L.max()) + 2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.hist(pix, bins=np.linspace(lo, hi, 80), color="#a67c52", edgecolor="none")
    ax1.axvline(pmean, color="k", ls="--")
    ax1.set_ylabel("Pixels")
    ax1.set_title(f"Pixel-level L*   mean={pmean:.1f}  CV={100 * pstd / pmean:.1f}%")

    ax2.hist(L, bins=np.linspace(lo, hi, 40), color="#6f4e37", edgecolor="white")
    ax2.axvline(mean, color="k", ls="--")
    ax2.set_xlabel("L* (lower = darker)")
    ax2.set_ylabel("Beans")
    ax2.set_title(f"Per-bean mean L*   n={len(L)}  mean={mean:.1f}  CV={100 * std / mean:.1f}%")

    fig.tight_layout()
    fig.savefig(path.with_name(path.stem + "_hist.png"), dpi=120)


if __name__ == "__main__":
    main()
