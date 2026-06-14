"""
The Modal Future — cover art generator.

Concept: a probability distribution over possible futures. The dominant mode —
the most-probable future — glows amber and is staked with a vertical marker and
a luminous point: the dated, falsifiable call the show is known for. The faint
secondary shoulder is the futures that didn't win. A measured baseline (ticks
like a timeline) reinforces "on the record, with dates."

Rendered at SS x final and downscaled with LANCZOS for clean anti-aliasing.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FINAL = 3000
SS = 2
R = FINAL * SS  # render resolution

def s(v):  # scale a final-space value into render space
    return int(round(v * SS))

# ---------------------------------------------------------------- palette
NAVY_TOP   = (12, 19, 35)
NAVY_BOT   = (8, 13, 26)
STEEL_FILL = (46, 64, 98)
AMBER      = (247, 167, 62)
AMBER_SOFT = (250, 186, 104)
AMBER_HOT  = (255, 224, 178)
WHITE      = (236, 240, 247)
MUTED      = (108, 124, 152)
GRID       = (38, 52, 78)

# ---------------------------------------------------------------- background
yy = np.linspace(0, 1, R)[:, None]
xx = np.linspace(0, 1, R)[None, :]
top = np.array(NAVY_TOP, float); bot = np.array(NAVY_BOT, float)
ycol = np.linspace(0, 1, R)[:, None]                 # (R,1)
vgrad = top[None, :] * (1 - ycol) + bot[None, :] * ycol   # (R,3) vertical gradient
bg = np.broadcast_to(vgrad[:, None, :], (R, R, 3)).copy()

# subtle radial lift behind the peak (cool), centred where the mode will sit
cx, cy = 0.40, 0.30
d = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 1.15) ** 2)
lift = np.clip(1 - d / 0.55, 0, 1) ** 2.2
bg += (np.array([26, 38, 62]) - np.array(NAVY_TOP)).clip(0) * lift[:, :, None] * 0.9

# vignette: darken corners
dv = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
vig = np.clip((dv - 0.45) / 0.45, 0, 1) ** 1.8
bg *= (1 - 0.35 * vig[:, :, None])

bg = np.clip(bg, 0, 255).astype(np.uint8)
img = Image.fromarray(bg, "RGB").convert("RGBA")

# ---------------------------------------------------------------- geometry
X0, X1 = 250, 2750            # curve horizontal extent (final space)
BASE   = 1545                 # baseline y (final space)
APEX_Y = 560                  # apex y of dominant mode (final space)
H      = BASE - APEX_Y        # pixel height of dominant mode

# distribution = dominant gaussian + lower, wider secondary shoulder
def dist(xf):
    t = (xf - X0) / (X1 - X0)            # 0..1 across the axis
    m1, s1, a1 = 0.34, 0.085, 1.00       # dominant mode
    m2, s2, a2 = 0.66, 0.150, 0.40       # secondary shoulder
    g1 = a1 * np.exp(-0.5 * ((t - m1) / s1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((t - m2) / s2) ** 2)
    return g1 + g2

XS = np.linspace(X0, X1, 1400)
norm = dist(XS).max()
YS = BASE - (dist(XS) / norm) * H
MODE_X = XS[np.argmax(dist(XS))]
MODE_Y = YS[np.argmin(YS)]

curve_pts = [(s(x), s(y)) for x, y in zip(XS, YS)]

# ---------------------------------------------------------------- fill under curve
# vertical gradient (steel near curve -> transparent at baseline), masked by the
# area under the curve, with a warm amber bias near the dominant mode column.
grad = np.zeros((R, R, 4), np.uint8)
gy = np.linspace(0, 1, R)[:, None]
ax_top, ax_base = s(APEX_Y), s(BASE)
span = ax_base - ax_top
rel = np.clip((np.arange(R)[:, None] - ax_top) / span, 0, 1)   # 0 at apex,1 at base
alpha = (1 - rel) ** 1.35 * 165
warm = np.clip(1 - np.abs(np.arange(R)[None, :] - s(MODE_X)) / s(560), 0, 1) ** 2
col = (np.array(STEEL_FILL)[None, None, :] * (1 - 0.45 * warm[:, :, None])
       + np.array(AMBER)[None, None, :] * (0.45 * warm[:, :, None]))
grad[:, :, :3] = np.clip(col, 0, 255).astype(np.uint8)
grad[:, :, 3] = (alpha * np.ones((1, R))).astype(np.uint8)
grad_img = Image.fromarray(grad, "RGBA")

mask = Image.new("L", (R, R), 0)
md = ImageDraw.Draw(mask)
poly = curve_pts + [(s(X1), s(BASE)), (s(X0), s(BASE))]
md.polygon(poly, fill=255)
img = Image.composite(grad_img, img, mask)

# ---------------------------------------------------------------- amber glow behind peak
glow = Image.new("RGBA", (R, R), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gr = s(330)
gd.ellipse([s(MODE_X) - gr, s(MODE_Y) - gr, s(MODE_X) + gr, s(MODE_Y) + gr],
           fill=AMBER + (110,))
glow = glow.filter(ImageFilter.GaussianBlur(s(120)))
img = Image.alpha_composite(img, glow)

draw = ImageDraw.Draw(img)

# ---------------------------------------------------------------- measured baseline
# faint timeline ticks + dashed axis
n_ticks = 13
for i in range(n_ticks):
    tx = X0 + (X1 - X0) * i / (n_ticks - 1)
    th = 30 if i % 3 == 0 else 16
    draw.line([(s(tx), s(BASE)), (s(tx), s(BASE + th))], fill=GRID + (255,), width=s(2.2))
# dashed baseline
dash, gap = 34, 26
x = X0
while x < X1:
    draw.line([(s(x), s(BASE)), (s(min(x + dash, X1)), s(BASE))],
              fill=MUTED + (140,), width=s(2.4))
    x += dash + gap

# ---------------------------------------------------------------- the curve line
# secondary part slightly dimmer than the dominant mode
def seg(a, b, color, w):
    draw.line([curve_pts[a:b]], fill=color, width=w, joint="curve")
draw.line(curve_pts, fill=(150, 170, 205, 255), width=s(5))     # base stroke
# brighten the dominant mode arc in amber
i_lo = np.argmax(XS > MODE_X - 250)
i_hi = np.argmax(XS > MODE_X + 250)
draw.line(curve_pts[i_lo:i_hi], fill=AMBER + (255,), width=s(7), joint="curve")

# ---------------------------------------------------------------- the mode marker
draw.line([(s(MODE_X), s(BASE)), (s(MODE_X), s(MODE_Y))], fill=AMBER + (255,), width=s(5))
# axis foot tick (emphasised)
draw.line([(s(MODE_X), s(BASE - 4)), (s(MODE_X), s(BASE + 46))], fill=AMBER + (255,), width=s(6))

# luminous point at the mode
dot_glow = Image.new("RGBA", (R, R), (0, 0, 0, 0))
dd = ImageDraw.Draw(dot_glow)
dr = s(95)
dd.ellipse([s(MODE_X) - dr, s(MODE_Y) - dr, s(MODE_X) + dr, s(MODE_Y) + dr], fill=AMBER_SOFT + (200,))
dot_glow = dot_glow.filter(ImageFilter.GaussianBlur(s(48)))
img = Image.alpha_composite(img, dot_glow)
draw = ImageDraw.Draw(img)
r1 = s(30)
draw.ellipse([s(MODE_X) - r1, s(MODE_Y) - r1, s(MODE_X) + r1, s(MODE_Y) + r1], fill=AMBER + (255,))
r2 = s(13)
draw.ellipse([s(MODE_X) - r2, s(MODE_Y) - r2, s(MODE_X) + r2, s(MODE_Y) + r2], fill=AMBER_HOT + (255,))

# ---------------------------------------------------------------- typography
FONTS = "C:/Windows/Fonts/"
def load(name, size, variation=None):
    f = ImageFont.truetype(FONTS + name, s(size))
    if variation:
        try: f.set_variation_by_name(variation)
        except Exception: pass
    return f

def text_tracked(cx, baseline_y, txt, font, fill, tracking):
    # measure
    widths = [draw.textlength(ch, font=font) for ch in txt]
    track = s(tracking)
    total = sum(widths) + track * (len(txt) - 1)
    x = s(cx) - total / 2
    for ch, w in zip(txt, widths):
        draw.text((x, s(baseline_y)), ch, font=font, fill=fill, anchor="ls")
        x += w + track

# wordmark
try:
    big = load("bahnschrift.ttf", 300, "SemiBold")
except Exception:
    big = load("seguibl.ttf", 290)
try:
    small = load("bahnschrift.ttf", 90, "SemiBold")
except Exception:
    small = load("seguisb.ttf", 90)

# "THE" — small, amber, widely tracked, on its own line well above the wordmark
text_tracked(1500, 2055, "THE", small, AMBER + (255,), 70)
# "MODAL FUTURE" — large, white, tight tracking
text_tracked(1500, 2390, "MODAL FUTURE", big, WHITE + (255,), 16)

# ---------------------------------------------------------------- finish
out = img.convert("RGB").resize((FINAL, FINAL), Image.LANCZOS)
out.save("cover_modal_future_v5.png")
print("wrote cover_modal_future_v5.png", out.size)
