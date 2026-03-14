## Session 11/02/2026

  * Meeting with Quentim Bammey and Christian Sandor (11/02/2026)

## Between 11/02/2026 and 20/02/2026

  * All :
    - Meeting with Quentim Bammey (19/02/2026)
    - Planning
  * Amayas :
    - Setting up with Ghiles a Makefile to easily configure and install the project.
      - Linking the GitHub repository of ComfyUI to our GitLab repository.
      - Writing a first version of Makefile that clones + compiles + starts the project.
    - Implementing V0 for black light substraction and reading raw images in ComfyUI.
    - Doing research about Malvar-He-Culter demosaicing algorithm and implementation.
  * Charlotte :
    - implementation of bilinear_demosaicing algorithm
    - familiarisation with ComfyUI, especially creation of nodes
  * Ghiles :
    - Setting up a Makefile with Amayas to easily configure and install the project.
      - Easy installation of ComfyUI dependencies.
      - Auto-detect OS and hardware to install the libraries accordingly.
    - Documentation on ComfyUI,raw image reading and demoisaicing.
    - Implementing Malvar He Cutter demosaicing algorithm in python. 
  * Rayane :

## Session 20/02/2026

  * Amayas :
    - Implemented a node for exposure compensation.
    - Added unitary tests for exposure compensation.
    - Created an issues board on GitLab and sorted issues with labels for better issues managing (To Do, V0, V1, V2, Bug, Done) and assigned issues for all group members.
  * Charlotte :
    - Configuration to connect remotely to another computer, more powerful than mine, so that ComfyUI runs faster, and installation of the necessary packages on that other computer.
    - First researches to understand "color manipulation" and the different features needed for this node (docs/color_manipulation.md)
    - Setting up issues managing on GitLab.
  * Ghiles :
    - Separated the black light substraction from the raw image reading and created two differents nodes for them.
    - Completed the CI/CD that Rayane started, with a prebuilt docker image to speed up the tests.
    - Refactored the architecture of the project to put put each algorithm in its apoprpiate folder that defines which phase of the ISP it belongs to.
    - Started researching the tone curve application algorithm. 
  * Rayane :

## Between 20/02/2026 and 23/02/2026

## Session 23/02/2026

  * Amayas :
    - Researching on color space transformation.
    - Researching on ways to extract `HueSatMap` LUT from camera metadata to perform tint correction on image. 
  * Charlotte :
    - Continue researches on color manipulation, especially adjusting Temperature
    - Tanner Helland Algorithm, converting Temperature in Kelvin to RGB
  * Ghiles :
    - Continued researching the tone curve application phase and way to easily edit the tone curve with a simple UI.
  * Rayane :

## Between 23/02/2026 and 13/03/2026
  * All
    - Meeting with Quentim Bammey (25/02/2026)
  * Amayas :
    - Worked with Rayane on solving PyTorch bugs related to image sizes when using quantile function on 12 Mpx or higher images and implementing nodes for white balance Rayane's algorithms.
    - Implemented $\text{HSV} \to \text{RGB}$ and $\text{RGB} \to \text{HSV}$ space transformations + their unitary tests.
    - Implemented reading `HueSatMap` LUT from `.dcp` files (necessary for hue/saturation fidelity stage).
  * Charlotte :
    - implementation Tanner-Helland Algorithm and temperature simple + creations of the nodes in Comfy UI
    - apply LUTs for color_manipulation:
      - researchs
      - import LUTs and read .cube file
      - first try: implementation of trilinear interpolation -> uses to much memory to work
      - so, implementation with torch.nn.functional.grid_sample() -> still some errors (questions to Amayas and Quentin Bammey about color spaces)
  * Ghiles :
  * Rayane :

## Session 13/03/2026

  * All :
    - Meeting with Quentin Bammey (13/03/2026)
    - Discussions regarding the schedule and the progress of our respective tasks
  
## Between 13/03/2026 and 16/03/2026

  * Amayas :
  * Charlotte :
    - correction of apply_lut_grid_sample():
      - change of color space (from linearRGB to AdobeRGB1998) to apply the luts
      - torch.clamp to remove black areas that should be white
  * Ghiles :
  * Rayane :