# Lanes (Lane Processing and Geometry Estimation)

This document specifies the lane mask processing, left/right edge extraction, median construction, and metric estimation (lateral offset and yaw angle ψ) designed for Jetson Nano with low compute, using linear fits x(y)=m·y+b and a row‑dependent scale s(y)=a·y+b derived from lane width at two ROI scanlines.

## Purpose and scope
- Detect left/right lane edges in a binary mask, build a robust centerline (median), and output near‑field lateral offset [m] and yaw angle ψ [rad] for the lateral controller, with deterministic timing on constrained hardware.

## Interfaces
- Input: single‑channel binary mask of lanes (selected model channel), resized to original frame, cropped to a vertical ROI between 50% and 95% of image height.  
- Outputs: annotated frame (debug), left/right line coefficients x(y)=m·y+b, median points, and LineIntersect {xlt, xlb, xrt, xrb, slope, ψ, offset}.

## Pre/post‑processing
- Threshold at 0.1 → binary mask.  
- Morphology: close + dilate (kernel 5×5).  
- Resize to original frame and zero pixels outside ROI (rows < 0.50·H or > 0.95·H).

## Edge extraction and validation
- Bottom‑up row scan from the image center: first white per row on left/right within bounded ranges; continuity counters and guarded resets to reject false detections near the center seam.  
- Validator codes: 0 (both sides valid), −1 (only right), −2 (only left), −3 (both lost). LaneMemory persists previous edges/coefficients for temporal robustness.

## Linear fit by image row y (per side)
For each side, a least‑squares line is fit as x(y)=m·y+b. In code, x is regressed on y (y is the independent variable), which is well‑conditioned for near‑vertical lane edges.  
Optionally, if needed for documentation, the closed‑form LS estimates with samples {(y_i, x_i)} are:
$$
m = \frac{n\sum y_i x_i - \left(\sum y_i\right)\left(\sum x_i\right)}{n\sum y_i^2 - \left(\sum y_i\right)^2}
\quad,\quad
b = \frac{\left(\sum x_i\right)\left(\sum y_i^2\right) - \left(\sum y_i\right)\left(\sum y_i x_i\right)}{n\sum y_i^2 - \left(\sum y_i\right)^2}
$$

Denote the fitted lines as:
$$x_{\text{left}}(y)=m_L\,y+b_L \quad,\quad x_{\text{right}}(y)=m_R\,y+b_R$$

The median (centerline) column at row y is:
$$x_m(y)=\tfrac{1}{2}\big(x_{\text{left}}(y)+x_{\text{right}}(y)\big)$$

## ROI scanlines and symbols
- y1 = top ROI row; y2 = bottom ROI row (y2 > y1).  
- xlt = x_left(y1), xrt = x_right(y1); xlb = x_left(y2), xrb = x_right(y2).  
- x_c = image center column (width/2).  
- W = lane width in meters (constant), here W = 0.26 m.  
- P1_x_car_frame, P2_x_car_frame = known forward distances (meters) from the vehicle’s CoM to rows y2 (bottom ROI) and y1 (top ROI), respectively.

## Row‑dependent pixel‑to‑meter scale s(y)
Define the pixel gap between lane edges at row y:
$$\text{width\_px}(y)=x_{\text{right}}(y)-x_{\text{left}}(y)$$

Enforce metric consistency using lane width W:
$$s(y)\cdot \text{width\_px}(y) = W \quad\Rightarrow\quad s(y)=\frac{W}{\text{width\_px}(y)}$$

Evaluate at the two ROI scanlines to obtain s(y1), s(y2):
$$s(y_1)=\frac{W}{\,x_{rt}-x_{lt}\,}\quad,\quad s(y_2)=\frac{W}{\,x_{rb}-x_{lb}\,}$$

Fit the linear scale model s(y)=a·y+b from the two points (y1,s(y1)) and (y2,s(y2)):
$$a = \frac{s(y_2)-s(y_1)}{y_2-y_1}\quad,\quad b = s(y_1) - a\cdot y_1$$

Thus, for any row y in the ROI:
$$s(y)=a\cdot y + b$$

## Lateral position in meters at row y
Convert median column to meters relative to the optical center x_c:
$$Y_{\text{car}}(y)=s(y)\cdot\big(x_m(y)-x_c\big)$$

In particular, at the two ROI scanlines:
$$Y_{\text{car}}(y_1)=s(y_1)\cdot\big(x_m(y_1)-x_c\big)\quad,\quad
Y_{\text{car}}(y_2)=s(y_2)\cdot\big(x_m(y_2)-x_c\big)$$

## Yaw (ψ) and near‑field offset
Using the known forward spacing between the two scanlines in the vehicle frame:
$$\Delta X_{\text{car}} = P2\_x\_{\text{car\_frame}} - P1\_x\_{\text{car\_frame}}$$
and the lateral change:
$$\Delta Y_{\text{car}} = Y_{\text{car}}(y_2) - Y_{\text{car}}(y_1)$$

Compute the centerline slope and yaw:
$$\text{slope}=\frac{\Delta Y_{\text{car}}}{\Delta X_{\text{car}}}\quad,\quad \psi=\arctan\big(\text{slope}\big)$$

Define the near‑field lateral offset (used by control) at the bottom ROI:
$$\text{offset}=Y_{\text{car}}(y_2)=s(y_2)\cdot\big(x_m(y_2)-x_c\big)$$

## Parameters and empirical tuning
- ROI: y ∈ [0.50·H, 0.95·H].  
- Morphology: kernel 5×5 (close + dilate).  
- Edge search: bounded horizontal ranges around last detected x with continuity counters; validator thresholds (e.g., min_size_line=50) tuned empirically.  
- Lane width: W = 0.26 m.  
- Longitudinal spacings: P1_x_car_frame (to bottom ROI), P2_x_car_frame (to top ROI) set from camera mounting and geometry.  
- Fallback memory: prior edges/coefficients reused under partial/total loss (codes −1/−2/−3) to stabilize the median and derived ψ/offset.

## Notes
- Linear x(y) fits and s(y)=a·y+b are chosen for deterministic, low‑compute runtime on Jetson Nano (no polynomial fits, no BEV), adequate for gentle to moderate curvature with the selected ROI.  
- The “LaneData” pixel‑to‑meter factors from earlier drafts are not used; the current metric conversion is fully defined by W, the two ROI scanlines, and P1/P2 constants.
