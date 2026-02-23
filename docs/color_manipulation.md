# Color Manipulation

## Description

Color manipulation is used to create a signature look and personalize image colors. It can be used to:

- Give a warmer or cooler look
- Change saturation
- Accentuate specific colors
- Create a vivid style (higher saturation, punchier colors)
- Create a portrait style (softer red/orange skin tones)
- Create a landscape style (enhanced blues/greens for sky, river, trees)
- Keep a standard style
- Keep a neutral style

## Difference from Hue/Sat Map

- Hue/Sat map focuses on color accuracy
- Color manipulation focuses on artistic or stylistic rendering

Therefore, the color manipulation node in ComfyUI should expose user adjustments so each user can customize color rendering based on their preferences.

## Features to Implement

- Temperature (blue/yellow)
- Tint (green/pink)
- Saturation
- Contrast

## Useful Python Libraries (HSV)

- kornia.color
- colorsys

## Temperature

### Simple Algorithm

- r = r + adjustment
- g = g
- b = b - adjustment
- Recommended adjustment range: [-20, 20]
- Full adjustment range: [-100, 100]

### Alternative Algorithm

- Tanner Helland algorithm (convert temperature in Kelvin to RGB)

## Tint

### Simple Algorithm

- r = r
- g = g + adjustment
- b = b
- Recommended adjustment range: [-20, 20]
- Full adjustment range: [-100, 100]