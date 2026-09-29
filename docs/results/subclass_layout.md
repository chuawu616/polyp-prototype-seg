# Layout of the foreground sub-classes (test sets, prototype similarity argmax)

### V0, original PPD (`runs/followup/v0_ppd_original/best.pth`)

798 test images; the dominant FG sub-class takes on average 0.53 of the polyp; > 90 % in 6% of images, > 70 % in 12%. Dominant sub-class vs. area: Kruskal-Wallis p = 8.7e-10; vs. brightness: p = 1.4e-06; NMI with the source dataset = 0.034.

| FG sub-class | images dominated | median polyp area | median brightness | depth, small polyps (pixel share) | depth, large polyps (pixel share) | height in large polyps | width in large polyps |
|---|---|---|---|---|---|---|---|
| FG0 | 43 | 0.041 | 0.556 | 0.39 (0.24) | 0.39 (0.24) | 0.63 | 0.32 |
| FG1 | 551 | 0.034 | 0.527 | 0.34 (0.41) | 0.32 (0.29) | 0.27 | 0.43 |
| FG2 | 204 | 0.084 | 0.460 | 0.31 (0.35) | 0.30 (0.47) | 0.62 | 0.65 |

Depth: 0 = boundary, 1 = centre. Height / width: 0 = top / left of the polyp bounding box; 111 large polyps (> 15% of the image).

### V0, corrected PPD (`runs/followup/v0_ppd_corrected/best.pth`)

798 test images; the dominant FG sub-class takes on average 0.58 of the polyp; > 90 % in 8% of images, > 70 % in 17%. Dominant sub-class vs. area: Kruskal-Wallis p = 2.7e-19; vs. brightness: p = 4.9e-06; NMI with the source dataset = 0.031.

| FG sub-class | images dominated | median polyp area | median brightness | depth, small polyps (pixel share) | depth, large polyps (pixel share) | height in large polyps | width in large polyps |
|---|---|---|---|---|---|---|---|
| FG0 | 318 | 0.046 | 0.484 | 0.17 (0.41) | 0.20 (0.52) | 0.50 | 0.56 |
| FG1 | 437 | 0.033 | 0.536 | 0.45 (0.41) | 0.33 (0.20) | 0.60 | 0.27 |
| FG2 | 43 | 0.234 | 0.477 | 0.52 (0.17) | 0.56 (0.28) | 0.48 | 0.57 |

Depth: 0 = boundary, 1 = centre. Height / width: 0 = top / left of the polyp bounding box; 111 large polyps (> 15% of the image).

### Pseudo, SP 50 px (`runs/followup/pseudo_sp50/best.pth`)

798 test images; the dominant FG sub-class takes on average 0.89 of the polyp; > 90 % in 72% of images, > 70 % in 81%. Dominant sub-class vs. area: Kruskal-Wallis p = 1.8e-29; vs. brightness: p = 6.8e-02; NMI with the source dataset = 0.029.

| FG sub-class | images dominated | median polyp area | median brightness | depth, small polyps (pixel share) | depth, large polyps (pixel share) | height in large polyps | width in large polyps |
|---|---|---|---|---|---|---|---|
| FG0 | 701 | 0.034 | 0.518 | 0.34 (0.85) | 0.25 (0.36) | 0.78 | 0.49 |
| FG1 | 71 | 0.258 | 0.477 | 0.21 (0.02) | 0.33 (0.43) | 0.29 | 0.49 |
| FG2 | 26 | 0.158 | 0.561 | 0.33 (0.13) | 0.46 (0.22) | 0.56 | 0.52 |

Depth: 0 = boundary, 1 = centre. Height / width: 0 = top / left of the polyp bounding box; 111 large polyps (> 15% of the image).

### Pseudo, SP 800 px (`runs/followup/pseudo_sp800/best.pth`)

798 test images; the dominant FG sub-class takes on average 0.88 of the polyp; > 90 % in 67% of images, > 70 % in 80%. Dominant sub-class vs. area: Kruskal-Wallis p = 3.5e-16; vs. brightness: p = 1.4e-06; NMI with the source dataset = 0.029.

| FG sub-class | images dominated | median polyp area | median brightness | depth, small polyps (pixel share) | depth, large polyps (pixel share) | height in large polyps | width in large polyps |
|---|---|---|---|---|---|---|---|
| FG0 | 67 | 0.241 | 0.491 | 0.26 (0.04) | 0.32 (0.41) | 0.34 | 0.50 |
| FG1 | 691 | 0.035 | 0.523 | 0.35 (0.88) | 0.31 (0.38) | 0.76 | 0.50 |
| FG2 | 40 | 0.008 | 0.384 | 0.28 (0.09) | 0.36 (0.21) | 0.43 | 0.48 |

Depth: 0 = boundary, 1 = centre. Height / width: 0 = top / left of the polyp bounding box; 111 large polyps (> 15% of the image).
