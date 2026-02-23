# Color manipulation

## Description

To have a signature, to personalize the colours of the image :
    - Give a warmer/cooler look
    - Change the saturation
    - Accentuate certain colours
    - Vivid (higher saturation, punchier colors)
    - Portrait (soften red/orange skin tones)
    - Landscape (enhanced blues/greens for the sky, river, trees...)
    - Standard
    - Neutral...

Difference with Hue/Sat map :
    - Hue/Sat map : color accuracy
    - Color manipulation : artistic or stylistic rendering

Therefore, the color manipulation node in ComfyUI should allow user adjustments, enabling them to customize the image’s colors according to their preferences.


Features to implement :
    - temperature (blue/yellow)
    - Tint (green/pink)
    - saturation
    - contrast

HSV Python libraries :
    - kornia.color
    - colorsys


Temperature :
    - simple algorithm :
        r = r + adjustment
        g = g
        b = b - adjustment
        recommended adjustment in [-20,20]
        adjustment in [-100, 100]
    - Tanner Helland Algorithm, converting Temperature in Kelvin to RGB

Tint :
    - simple algorithm :
        r = r 
        g = g + adjustment
        b = b
        recommended adjustment in [-20,20]
        adjustement in [-100, 100]