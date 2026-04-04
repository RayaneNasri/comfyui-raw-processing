# ComfyUI Nodes & Image Processing API Reference

## Table of Contents

(after exposure compensation)
7. [Color Manipulation Nodes](#color-manipulation-nodes)
   - [Temperature Simple](#temperature-simple)
   - [Temperature Tanner-Helland](#temperature-tanner-helland)
   - [LUT Color manipulation](#lut-color-manipulation)


## Overview

### Pipeline Architecture

After Exposure Compensation : Color Manipulation

### Data Types & Transformations

### Key concepts

(after Exposure)
- **Color Manipulation:** Customize the image colours

## Color Manipulation Nodes

Color adjustments allow users to customize the colors in an image. They can use a lookup table (LUT) to apply a preset style to their image, or (and) manually adjust the image's temperature and saturation using specific nodes.

### Overview: Color Manipulation features

| Method | Input | Use Case |
|--------|-------|----------|
| **Temperature Simple** | Adjustement choosen by the user | To enhance blue or red simply |
| **Temperature Tanner-Helland** | Adjustement choosen by the user | Give a warmer/cooler look |
| **LUT Color Manipulation** | LUT : choosen in a list, or user-provided .cube file | Apply a preset style |

### Temperature Simple

**Description**

Enhance blue or red by simply adding the adjustement value for the red channel and removing it for the blue channel : not very realistic, but simple and fast.

**Status:** Automatic with one parameter
**Category:** image/processing
**Outputs:** 1

#### ComfyUI Interface

**Node Name:** `Temperature Simple`  
**Category:** image/processing

TODO: add image

#### Input Parameters

| Input | Type | Range | Default |
|-------|-------|-------|-------|
| **rgb_image** | Tensor - shape : (H, W, 3) - Dtype : float32 | [0, 1] | — |
| **adjustement** | float | [-100, 100] (recommended [-20, 20]) | 0 |

#### Output Schema

| Output | Type | Range |
|--------|-------|-------|
| **rgb_image** | Tensor - shape : (H, W, 3) - Dtype : float32 | [0, 1] |

#### Algorithm Details

1. Normalize the adjustement value (/255)
2. Given a temperature adjustment on the range -100 to 100 (recommended -20, 20), apply the following adjustment to each pixel in the rgb_image :
   red = red + adjustment_value
   green = green
   blue = blue - adjustment_value
3. Clamp values to the range [0, 1]

#### ComfyUI Usage

**Step-by-step:**

1. Connect **rgb_image** from Exposure Compensation node
2. Adjust the parameter (recommended in [-20, 20])
3. Connect output to Tone Curve Application node

**When to use:**
- to apply fast and easy change on the image temperature

**When NOT to use:**
- Professional color correction needed
- White balancing needs to be preserved

### Temperature Tanner-Helland

**Description**

Give a warmer/cooler look to the image. Given a temperature in Kelvin (representing a type of light, warm or cool), estimates an RGB equivlent (algorithm based on empirical values), and apply it to the image.

**Status:** Automatic with one parameter 
**Category:** image/processing
**Outputs:** 1

#### ComfyUI Interface

**Node Name:** `Temperature Tanner-Helland`  
**Category:** image/processing

TODO: add image

#### Input Parameters

| Input | Type | Range | Default |
|-------|-------|-------|-------|
| **rgb_image** | Tensor - shape : (H, W, 3) - Dtype : float32 | [0, 1] | — |
| **temperature_Kelvin** | float | [1000, 40000] (recommended [1500, 15000]) | 6600 |

#### Output Schema

| Output | Type | Range |
|--------|-------|-------|
| **rgb_image** | Tensor - shape : (H, W, 3) - Dtype : float32 | [0, 1] |

#### Algorithm Details

1. Convert the target temperature in Kelvin to RGB using the tanner Helland algorithm : a simple, straightforward algorithm, based on empirical values (https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html)
2. Normalize the three coefficients (red, green, blue)
3. Multiply each pixel by the coefficients on the corresponding colour channel

#### ComfyUI Usage

**Step-by-step:**

1. Connect **image** from Exposure Compensation Node
2. Adjust the temperature parameter (recommended [1500, 15000])
3. Connect output to Tone Curve Application node

**When to use:**
- to give a warmer or cooler look to the image

**When NOT to use:**
- Professional color correction needed
- White balancing needs to be preserved

### LUT Color Manipulation

**Description**

Use a Look-Up Table (LUT) to apply a preset style to the image. The LUT can be choosen in a list, or it can be provided by the user via a .cube file, the LUT color space being AdobeRGB1998.

**Status:** Automatic with one parameter
**Category:** image/processing
**Outputs:** 1

#### ComfyUI Interface

**Node Name:** `LUT Color Manipulation`  
**Category:** image/processing

TODO: add image

#### Input Parameters

| Input | Type | Range | Description |
|-------|-------|-------|-------|
| **rgb_image** | Tensor - shape : (H, W, 3) - Dtype : float32 | [0, 1] | Image on which to apply the LUT |
| **lut_name** | String | Choosen in a list | Selection of the title of a LUT (representing a key in a dictionnary, corresponding value being the path for the LUT) |
| **apply_lut_from_lut_path** | Boolean | - | True : apply the lut given by "lut_path" / False : apply the lut choosen in the list, ie "lut_name" |
| **lut_path** | String | - | A path to a LUT (color space AdobeRGB1998) in a .cube file |

#### Output Schema

| Input | Type | Range | Description |
|-------|-------|-------|-------|
| **rgb_image** | Tensor - shape : (H, W, 3) - Dtype : float32 | [0, 1] | Image on which to apply the LUT |

#### Algorithm Details

1. Read the .cube file (given by the user, or corresponding to the one selected in the list by the user)
   - read and save the size of the LUT (written in the file next to the expression LUT_3D_SIZE)
   - for each line with 3 values, save the line in a list
   - create the LUT, a tensor with the data given by the list, dtype float32, with dimension (size, size, size, 3)

2. Converts the image from LinearRGB to AdobeRGB1998.
   - AdobeRGB1998 values apply gamma correction to LinearRGB values using a simple power function:
      v = u^ɣ,          u ≥ 0
      v = -(-u)^ɣ,      u < 0
      with ɣ = 1/2.19921875

3. Apply the LUT via torch.nn.functional.grid_sample(lut, grid, mode='bilinear', align_corners=True)
   - "lut" is the LUT transformed as a (1,3,size,size,size) Tensor
   - "grid" is the rgb_image transformed in range [-1, 1], with color channels order being BGR (and not RGB), and with dimension (B,1,H,W,3) 
   - "bilinear" is to have a trilinear interpolation

4. Change the output to correspond to the original format of rgb_image 

#### ComfyUI Usage

**Step-by-step:**

1. Connect **image** from Exposure Compensation node
2. Choose a LUT in the list, or write the path to a .cube file representing a LUT and in that case select True for "apply_lut_from_lut_path"
3. Connect output to Tone Curve Application node

**When to use:**
- apply a style to the image

**When NOT to use:**
- want to stay close to the original photo / to reality

## Python API Usage (without ComfyUI)

TODO : complete and correct this section

from algorithms.color_manipulation._temperature_simple import temperature_simple
from algorithms.color_manipulation._temperature_tanner_helland import temperature_tanner_helland
from algorithms.color_manipulation._lut_color_manipulation import load_cube_lut
from algorithms.color_manipulation._lut_color_manipulation import apply_lut_grid_sample

Step 6: Color Manipulation (optional)
print("Step 6: Color Manipulation...")
col_img = lut_color_manipulation(exp_img, ...)

Step 7: Gamma Correction
... col_img ...


## Best Practices

## Appendix

### Glossary

- **LUT:** Look-Up Table

### External References