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
  * Rayane:
    - Started working on the implementation tasks assigned by the professor during the meeting:
      - Loading raw images
      - Converting raw images to RGB format
      - Preparing bilinear demosaicing tests for Charlotte

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
  * Rayane:
    - Learning CI/CD concepts and implementing the first version

## Between 20/02/2026 and 23/02/2026
  * Rayane:
    - Implemented the White Patch reference algorithm
    - Implemented the Gray World algorithm
    - Developed associated tests

## Session 23/02/2026

  * Amayas :
    - Researching on color space transformation.
    - Researching on ways to extract `HueSatMap` LUT from camera metadata to perform tint correction on image. 
  * Charlotte :
    - Continue researches on color manipulation, especially adjusting Temperature
    - Tanner Helland Algorithm, converting Temperature in Kelvin to RGB
  * Ghiles :
    - Continued researching the tone curve application phase and way to easily edit the tone curve with a simple UI.
  * Rayane:
    - Conducted research on gamma curve correction phase

## Between 23/02/2026 and 13/03/2026
  * All
    - Meeting with Quentin Bammey (25/02/2026)
  * Amayas :
    - Worked with Rayane on solving PyTorch bugs related to image sizes when using quantile function on 12 Mpx or higher images and implementing nodes for white balance Rayane's algorithms.
    - Implemented $\text{HSV} \to \text{RGB}$ and $\text{RGB} \to \text{HSV}$ space transformations + their unitary tests.
    - Implemented reading `HueSatMap` LUT from `.dcp` files (necessary for hue/saturation fidelity stage).
    - Implemented firs version of function applying an HSV LUT to an RGB image (deprecated later).
    - Started reading the Adobe DNG Specifications document to understand how to use `.dcp` data to perform hsv correction.
    - Implemented an early version of the hue, saturation and value mapping algorithm that uses `.dcp` data : `LUT1`, `LUT2`, `ColorMatrix1`, `ColorMatrix2`, `ForwardMatrix1`, `ForwardMatrix2` and illuminant values.
    - Wrote unitary tests for hue, saturation and mapping algorithm.
    - Fixed some refactoring and CI/CD issues for branches `feature/gamma-correction` and `feature/noise-reduction`. 
  * Charlotte :
    - implementation Tanner-Helland Algorithm and temperature simple + creations of the nodes in Comfy UI
    - apply LUTs for color_manipulation:
      - researchs
      - import LUTs and read .cube file
      - first try: implementation of trilinear interpolation -> uses to much memory to work
      - so, implementation with torch.nn.functional.grid_sample() -> still some errors (questions to Amayas and Quentin Bammey about color spaces)
  * Ghiles :
    - Implemented a first version of the tone curve application node in ComfyUI.
    - Used the same Adobe DNG specifications document as Amayas to extract the tone curve profile from the `.dcp` data and apply it to the image.
    - Tried to implement white balance algorithm that uses the wb gains of the camera, but it seems that the results are not good (surely because of my implemtation).
    - Added a comparison node of white balance algorithms in ComfyUI to compare the results of the different algorithms. 
  * Rayane:
    - Implemented the gamma correction phase along with its associated tests
    - Developed wrappers for OpenCV denoising functions
    - Implemented the denoising phase and its related tests

## Session 13/03/2026

  * All :
    - Meeting with Quentin Bammey (13/03/2026)
    - Discussions regarding the schedule and the progress of our respective tasks. 
  
## Between 13/03/2026 and 16/03/2026

  * Amayas :
    - Worked on merging features, adding issues to kanban and tasks organizing.
  * Charlotte :
    - correction of apply_lut_grid_sample():
      - change of color space (from linearRGB to AdobeRGB1998) to apply the luts
      - torch.clamp to remove black areas that should be white
    - modification of the ComfyUI node : select a lut in a list or import one
  * Ghiles :
    - Nothing worth mentioning.
  * Rayane :
    - Nothing worth mentioning.

## Session 16/03/2026

  * Amayas :
    - Did research about specifications of the hue, saturation and value mapping algorithm described in the Adobe DNG Specifications. Specially, about applying it on already white balanced and linear RGB images.
  * Charlotte :
    - start of research into deblurring
    - reading of "A Simple Local Minimal Intensity Prior and An Improved Algorithm for Blind Image Deblurring" - Fei Wen, Rendong Ying, Yipeng Liu, Peilin Liu, and Trieu-Kien Truong
  * Ghiles :
    - Nothing worth mentioning.
  * Rayane:
    - Nothing worth mentioning.

## Between 16/03/2026 and 27/03/2026

  * Amayas :
    - Adapted the hue, saturation and value mapping algorithm to match the result expected on linear RGB images that are already white balanced, deleted the linearization part, now the algorithm only requires these `.dcp` data: `ColorMatrix1`, `ColorMatrix2`, `LUT1`, `LUT2` and calibration illuminants.
    - Adapted the unitary tests to the new version of the hue, saturation and value mapping algorithm. 
    - Added a merge request to merge the hue, saturation and value mapping node feature into dev and label it as a V0 feature.
    - Wrote documentation for first release with Ghiles
  * Charlotte :
    - Change of plan : I started reading the matlab code for the "Blind Deblurring - PMP - Fei Wen" algorithm, but I decided it would be easier and faster to first adapt the implementation of the Goldstein-Fattal method made by Said Ladjal
    - Re-reading (first read during the UE intersemestre "Problèmes en imagerie computationnelle") of the slides "Flou dans les images" made by Saïd Ladjal and the article "Estimating an Image’s Blur Kernel Using Natural Image Statistics, and Deblurring it: An Analysis of the Goldstein-Fattal Method - Jérémy Anger, Gabriele Facciolo, Mauricio Delbracio" 
  * Ghiles :
    - Added documentation with amayas of the project in the `docs/` folder of the repository.
  * Rayane:
    - Nothing worth mentioning.
  
## Session 27/03/2026

  * All :
    - Meeting with Quentin Bammey (27/03/2026)
  * Amayas :
    - Completed writing documentation for the release.
    - Worked with Ghiles on merging the release to the main branch.
    - Worked with Ghiles on resolving merge conflicts.
  * Charlotte :
    - Implementing ComfyUInode and main function for Goldstein-Fattal algorithm
  * Ghiles :
    - Talked with Quentin Bammey about making a more intuitive editor for tone curve and started looking up how to do it.
  * Rayane :
    - Start research about denoising deep learning algorithms

## Between 27/03/2026 and 03/04/2026
  * Amayas :
    - Worked on identifying the causes of the pipeline's excessive memory consumption.
    - Did research on in-place operations and memory manipulation with PyTorch.
    - Got back to and old version of the HSV Mapping due to an issue of the final color accuracy checking and started refactoring code.
  * Charlotte :
  * Ghiles :
    - Found a repository that implements an editor for color manipulation.
    - Dove deeper into the code of this repository to understand how it works and how comfyui nodes are communicating with the javascript code of the editor.
  * Rayane :
      Research about denoising deep learning algorithms
## Session 03/04/2026
  * All :
    - Meeting with Quentin Bammey (03/04/2026)

  * Amayas :
    - Implemented first tests to optimize memory in the pipeline, tried to remove the maximum of operations that could duplicate image tensors during the execution.
    - Results were effective but the pipeline still consumes a decent amount of memory (> 8 Go)
  * Charlotte :
    - Documentation for the color_manipulation node
  * Ghiles :
    - Implemented a first version of a curve editor with polynomial interpolation. It is not really user-friendly yet, but it is a first step.
  * Rayane :
      - Continue research about denoising deep learning algorithms

## Between 03/04/2026 and 10/04/2026

  * Amayas :
    - Continued research on optimizing memory to find a solution to minimise the number of intermediate tensors used.
    - Finalized the HSV mapping algorithm.
    - Added a new gamma correction node "IEC Gamma Correction" which happened to have more accurate contrasts.
    - Added tests for the IEC Gamma Correction. 
  * Charlotte :
    - implementation of saturation_hsv
    - correction in _lut_color_manipulation.py (output dimension in apply_lut_grid_sample())
  * Ghiles :
    - Researched about lens correction especially vignetting. I found some papers about these topics and I will try to implement them in the next weeks.
  * Rayane :
    - Start working on masking feature to process differently multiple parts of an image

## Session 10/04/2026

  * All :
    - Meeting with Quentin Bammey (10/04/2026)
  * Amayas :
  * Charlotte :
  * Ghiles :
    - Talked with Quentin Bammey about the lens correction phase and he gave additional phases for it like chromatic aberration correction and geometric distortion correction. I will try to implement these different corrections in the next weeks.
    - Started on a first version of vignetting (but no node yet in ComfyUI).
  * Rayane :
    - Discuss the masking feature with Quentin Bammey

## Between 10/04/2026 and 15/04/2026

  * Amayas :
  * Charlotte :
    - implemented script_test_luts and tested all the ON1 LUTs I have
    - correction in _lut_color_manipulation.py (final correction bgr -> rgb in apply_lut_grid_sample())
  * Ghiles :
    - Implemented chromatic aberration correction
    - Implemented geometric distortion correction
    - Made a node that aggregates the different lens corrections.
  * Rayane :
    - Start implementation of the masking feature
    - Working on the merge of `feature/noise-reduction` branch

## Session 15/04/2026

  * All :
    - Meeting with Quentin Bammey (15/04/2026)
  * Amayas :
  * Charlotte :
  * Ghiles :
    - spent the remaining TH, debugging a problem of  the lens correction wherre it was using a lot of ressources and was very slow.
    - After debugging, I found out that the problem was coming from the geometric distortion correction part, and more specifically from the way I was applying the distortion to the image. I was using a for loop to apply the distortion to each pixel of the image, which was very slow. I changed it to a vectorized implementation using PyTorch's `grid_sample` function, and now it is much faster but then switched to openCV's `undistort` function which looked faster and more efficient. 
  * Rayane :

## Between 15/04/2026 and 21/04/2026

  * Amayas :
    - Started to identify the main components of the pipeline that caused the excessive memory comsumption.
    - Ran time and memory profilers and analyzed the reports to identify which operations of the source code needed a lot of memory and ran slow.
  * Charlotte :
  * Ghiles :
    - Spent a bit of time looking what causes the pipepline to be slow and trying to optimize it.
    - I had some clues but didn't have time to try and implement them.

## Session 21/04/2026

  * Amayas :
    - Worked on project management : issues, planning and updating myself on the difficulties of other members.
  * Charlotte :
  * Ghiles :
    - Mainly nothing worth mentioning because of the exams.

## Between 21/04/2026 and 05/05/2026

  * All : 
    - Meeting with Quentin Bammey (24/04/2026)
    - Discussions regarding the schedule and the progress of our respective tasks.
  * Amayas :
    - Merged the final and official version of the HSV Mapping and IEC Gamma Correction.
  * Charlotte :
    - v0 Stakes Report (RAPPORT_ENJEUX.md)
  * Ghiles :
    - Continued improving the curve editor, mainly changing the interpolation method to  have a better control of the curve.
  * Rayane :
    - Making denoising nodes more robust by implementing a better error handler 

## Session 05/05/2026

  * All : 
    - Meeting with Quentin Bammey (05/05/2026)
    - Discussion on the stakes report
  * Amayas :
  * Charlotte :
    - Searching some sources/references for the stakes report
  * Ghiles :
    - Looked into bezier curves and started sketching a bezier curve editor, i will try to implement it in the next weeks.
  * Rayane :
    - Continue work on denoising nodes
    - Help Amayas optimizing code
    
## Between 05/05/2026 and 13/05/2026

  * Amayas :
    - Refactored the pipeline source code to delete all the intemerdiate tensors that are not being used during the execution of an algorithm.
    - Refactored how some masks are applied into image tensors to reduce the memory + execution time.
    - Refactored the code to use only in-place operations for updating tensors.
    - Ran memory and time profilers and noticed a decent gain on memory usage, now the pipeline uses on average 3 or 4 Gb of RAM.
    - Merged the `fix/memory-usage` branch that fixes the memory performance into `dev`.
  * Charlotte :
  * Ghiles :
    - Researched about classical methods for masking and segmentation of images.
  * Rayane :
    - Bibliography research about masking with deep learning algorithms

## Session 13/05/2026

  * Amayas :
  * Charlotte :
    - changing code in color-manipulation algorithms and nodes to have the initial squeeze and final unsqueeze of the batch (of images coming from ComfyUI) in the nodes, and not in the algorithms.
    - research on contrast algorithms
    - implementing contrast_linear_global and testing it with several contrast_factors
  * Ghiles :
    - Fixed a bug in the makefile where it it wasn't removing the nodes of other branches in the submodule of ComfyUI, which was causing problems when switching branches.
  * Rayane :
    - Continue research about masking with deep learning algorithms

## Between 13/05/2026 and 27/05/2026

  * All :
    - Meeting with Quentin Bammey (22/05/2026)
  * Amayas :
  * Charlotte :
    - Implementation of contrast_clahe (with kornia and with cv2), but the result does not look good on RGB images (it is initially designed for grayscale images)...
    - Preparation of the presentation for the audit P2P
  * Ghiles :
    - Implemented a first version of a bezier curve editor, it is not really user-friendly yet, but it is a first step, i let amayas take care of it.
  * Rayane :
    Nothing worth mentioning.
## Session 27/05/2026

  * All :
    - Audit P2P

## Between 27/05/2026 and 10/06/2026

  * All :
    - Meeting with Quentin Bammey (29/05/2026)
  * Amayas :
  * Charlotte :
    - Deblurring : corrected some errors on the main function of the Goldstein-Fattal algorithm, and tested it on an image
  * Ghiles :
    - Didn't work a lot because of the exams.
    - Refined a node for the curve editor.
    - Searched up how to do final color space conversion.
  * Rayane :
    Nothing worth mentioning because of the exams.
## Session 10/06/2026

  * Amayas :
    - Helped Charlotte for the memory check in color manipulation algorithms
    - Poster
  * Charlotte :
    - Memory check in color manipulation algorithms with Amayas
  * Ghiles :
    - Read a paper about HDR with Rayane Nasri , took more time that expected to understand:
      - the different HDR techniques : aligning the images, merging them and applying tone mapping
      - the different tone mapping algorithms : global and local tone mapping
      - Looked around in the internet to find datasets of images for HDR, but didn't find any that was really useful for our project since HDR requires multiple images of the same scene with different exposures, and most datasets only have one image per scene.
  * Rayane :
    - Read a paper about HDR with Ghiles Maloum

## Session 15/06/2026

  * All :
    - Meeting with Quentin Bammey (15/06/2026)
  * Amayas :
  * Charlotte :
    - Implemented read_image
    - Implemented read_image_node and deblurring_goldstein_fattal_node for ComfyUI and tested it
  * Ghiles :
    - Implemented a mock version of a classic HDR algorithm in normal python instead of ComfyUI; i basically used numpy and OpenCV to implement the different steps of the HDR algorithm, and then tested it on a few images.
      - The results were okayish, but not so great because i didn't have a lot of images with different exposures.
      - A problem was that it took a lot of time to process the images, mainly because of the alignment step, and also because it's done in numpy and OpenCV instead of PyTorch, which is much faster.
    - After the meeting with Quentin Bammey, he suggested to look into HDR+ algorithm and gave us an IPOl link of it. After searching with Rayane, we found the github repo of his code and we will try to understand it and implement it in ComfyUI.
  * Rayane :
    - Familiarize with the github repo of the HDR paper's author with Ghiles Maloum

## 22/06/2026

  * All :
    - Meeting with Quentin Bammey (22/06/2026)
  * Amayas :
  * Charlotte :
    - Added tests and little modifications in the documentation for temperature_simple, temperature_tanner_helland, saturation_hsv and contrast_linear_global
    - Changed the node and implementation of apply_lut to take into account the color_spaces (linearRGB or adobeRGB1998) of the image and of the lut
  * Ghiles :
    - Started implementing the HDR+ algorithm in matplotlib and numpy before doing it in comfyui.
    - I basically followed the same code as the github repo of the HDR+ paper's author, i just rewrote it in a more readable way and also took into consideration the newer packages because the repo was from 5 years ago.
    - The results were quite convincing, but it's still slow to ship.
  * Rayane :
    - Merging denoising branch into dev and fixing the related issues
    - Implement a ComfyUI node for exportation of images in different formats (png, jpg, tiff, etc.) and with different compression levels

## 23/06/2026

  * Amayas :
  * Charlotte :
    - lut in RGB or BGR: changed the node and implementation of apply_lut to take into account the order of the color channels of the lut (RGB or BGR) + separation of the node for personal/non-personal lut
    - formatting, linting, type checking and then merging of feature/color-manipulation
    - formatting, linting, type checking and then merging of feature/deblurring
  * Ghiles :
    - Reimplemented the multiple files of the HDR+ function by function in pytorch because there was no easy way to do it directly (the code in the github repo is poorly written)
    - Adding some documentation on some part of the code where it wasn't done.
  * Rayane :
    - Help Charlotte to merge the feature/deblurring branch into dev and fix the related conflicts
    - Standardize the input and output of the nodes
    - Cleaning the code and repo

## 24/06/2026

  * All :
    - Meeting with Quentin Bammey (24/06/2026)
  * Amayas :
  * Charlotte :
    - Implemented ComfyUI node for contrast_linear_global
    - Documentation for all the ComfyUI nodes I made
  * Ghiles :
    - After testing the code of yesterday, i found out that it required large amounts of VRAM to run, so basically it wasn't easy to just make the equivalent numpy + numba ==> pytorch, so i started to look with Rayane for a better way to optimize it. After speding hours in debugging and nitpicking the code, we couldn't just run. We decided that it was better to focus on finalizing other things in the project, so we ditched HDR+ for now and we will come back to it later if we have time.  
  * Rayane :
    - Join Ghiles to help him with the HDR+ algorithm in ComfyUI
    - Writing the documentation and put it in the same format for all nodes
    
## 25/06/2026

  * All :
    - Meeting with Quentin Bammey: Final Presentation (25/06/2026)
    - Cleaning up the rest of the code
    - Preparing pipelines to showcase tomorrow
    - Structuring and preparing the final presentation for tomorrow
