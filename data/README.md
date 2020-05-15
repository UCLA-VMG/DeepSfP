# **Deep Shape from Polarization: Dataset**

- This README refers to the published [SfP Dataset](https://drive.google.com/file/d/1EtjfMTfpanJotH92GFz300X_ZEmGXuqr/view). See instructions [here](https://github.com/alexrgilbert/deepsfp/blob/master/README.md) for usage with the published [SfP Codebase](https://github.com/alexrgilbert/deepsfp).
- Data was collected by the team at PKU, and prepared by the team at UCLA.

## **Data Summary:**

In total, the dataset contains 33 objects. 26 of the objects are reserved for
the train set used in the paper, while the remaining 7 are used for the test
set (for which results were presented in the [main paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690545.pdf), as well as the [supplement](https://visual.ee.ucla.edu/assets/deepsfp_eccv_supp.pdf)). (See enclosed train_list.csv and test_list.csv for the exact train and test
splits respectively. The test_list_supp.csv file contains additional orientations of the test objects from which results in the supplement were drawn.).


## **Directory Structure**


For the majority of objects (those with asymmetries for different views),
data was captured with the object in 4 different orientations:
front (f), back (b), left (l), and right (r). Note, some objects were only captured in a single orientation due to their inherent symmetries (and thus don't contain an orientation indicator in their filename). Additionally, each of these was repeated under 3 different lighting conditions:
indoors, outdoors on a sunny day, and outdoors on a cloudy day. After
curating the set to remove similarities between the train and test splits, the
final sample counts for the two splits are 236/21. The objects can be found in the sub-directory of the `surface_normals` dataset directory called `objects`:


````bash
DATADIR/
├── README.md
├── SurfaceNormals/
|   ├── train_list.csv
|   ├── test_list.csv
|   ├── test_list_supp.csv
|   └── objects
│       ├── <lighting_condition>/
│           ├── <item_name>_<orientation>.mat
│           └──             ...
│       └──              ...
````

## **Data Format**

Each data sample has 3 components: the polarization images, associated ground truth normals, and a binary foreground-mask indicating which pixels belong to the object. These are stored together in a single Matlab file, which can be loaded as a dictionary with 3 entries:


        1 - images [1024x1224x4 double] = The polarization images, where each channel
        corresponds to a different angle of the polarizer, specifically 0° for the 0th channel, 45° for the 1st channel, 90° for the 2nd channel, and 135° for the 4th channel.

        2 - mask [1024x1224 uint8] = The foreground mask for each object. Pixels equal to 1 correspond to pixels in the images and normals belonging to the object, while 0 pixels contain only background.

        3 - normals_gt [1024x1224x3 double] = The ground truth surface normals for
        each object. The channels correspond to X (0th channel), Y (1st channel), and Z (2nd channel) components of the normals
        at each pixel. The collection process details for these annotations is described below.

## **Data Visualization Tool**

We provide a tool [here](https://github.com/alexrgilbert/deepsfp/blob/master/tools/data_visualizer.ipynb) for visualizing samples in our polar image / surface normal dataset.

## **Collection Methods**


    Polarization images were captured using a Phoenix 5.0 MP Polarization camera
    from Lucid Vision Labs (https://thinklucid.com/product/phoenix-5-0-mp-polarized-model/).

    Ground truth surface normals were obtained through a 3 step process:

        1 - Obtain high-quality 3D shapes using a SHINING 3D scanner
        (https://www.einscan.com/desktop-3d-scanners/einscan-sp/), with single shot
        accuracy no more than 0.1mm, point distance from 0.17 mm to 0.2 mm, and a
        synchronized turntable for automatically registering scanning from multiple
        viewpoints.

        2 - Align scanned 3D mesh's from the scanner’s coordinate system to the
        image coordinate system of the polarization camera using 3D modeling tool
        MeshLab (https://www.meshlab.net/).

        3 — Compute surface normals using the Mitsuba renderer
        (http://www.mitsuba-renderer.org/).


## **Calculating Priors for Reconstruction**

Key to our method is the calculation of reconstruction priors for the surface normals of each object, satisfying  the classical equations of 'shape from polarization'. The details of the calculation process are explained extensively in Section 1 of the [supplement](https://visual.ee.ucla.edu/assets/deepsfp_eccv_supp.pdf) to our ECCV submission.

Before running our method on the dataset described above, an additional (4th) field should be added to each item called `normals_prior`, which is a 1024x1224x9 double. For each pixel, the first three channels correspond to the 'diffuse' solution, i.e. estimate of the X,Y, and Z of the pixel's normal vector as if it's reflection were predominantly diffuse. Similarly, the 4rd-6th and 7th-9th channels correspond to the two potential 'specular' solutions, respectively. Note, we perform these calculations using Matlab. A brief summary of the relevant equations is provided below, refer to the [supplement](https://visual.ee.ucla.edu/assets/deepsfp_eccv_supp.pdf) for more details.

### **Preliminaries**

- Refractive index, *n*: Assumed to be ~ 1.5

- Degree of polarization (DoP):
$\rho = \frac{I_{max} - I_{min}}{I_{max} + I_{min}}$

- Unpolarized Intensity: $I_{un} = \frac{I_{max} + I_{min}}{2}$

- Polarized Intensity for $\phi_{pol}$: $I(\phi_{pol}) = I_{un} + I_{un}\rho\cos(2\phi_{pol})\cos(2\phi) + I_{un}\rho\sin(2\phi_{pol})\sin(2\phi)$

### **Solving for $\rho$ and $\phi$**

- $a = I_{un}\cdot\rho cos(2\phi)$, $b = I_{un}\cdot\rho sin(2\phi)$
$\begin{bmatrix}
    1 & cos(2\phi_{pol,1}) & sin(2\phi_{pol,1}) \\
    &\vdots& \\
    1 & cos(2\phi_{pol,m}) & sin(2\phi_{pol,m})
    \end{bmatrix}
    \begin{bmatrix}
    I_{un} \\
    a \\
    b
    \end{bmatrix}
=    
    \begin{bmatrix}
    I_{\phi_{pol,1}} \\
    \vdots \\
    I_{\phi_{pol,m}}
    \end{bmatrix}$

- DoP ($\rho$) and phase ($\phi$):  $\rho = \frac{\sqrt{a^2 + b^2}}{I_{un}}\qquad
 \phi = \frac{1}{2}atan2(b,a)$

### **Diffuse solution**

#### **Azimuth ($\varphi$) and Zenith ($\theta$)**

$\varphi = \phi$

$\rho = \frac{(n - \frac{1}{n})^2 \sin^2\theta}{2 + 2n^2 - (n + \frac{1}{n})^2 \sin^2\theta + 4\cos\theta \sqrt{n^2 - \sin^2\theta}}$

(The above equation can be rearranged to obtain a closed form solution for $\theta$)

### **Specular solutions**

#### **Azimuth ($\varphi$) and Zenith ($\theta$)**

$\varphi = \phi + \frac{\pi}{2}$

$\rho = \frac{2 \sin^2\theta \cos\theta \sqrt{n^2 - \sin^2\theta}}{n^2 - \sin^2\theta - n^2\sin^2\theta + 2\sin^4\theta}$

(The above equation has no analytical solution. We thus produce two possible values for $\theta$ using numerical interpolation, yielding $N_{spec,1}$ and $N_{spec,2}$, respectively)

### **Calculating Normals**

$N = \begin{pmatrix}cos(\varphi) sin(\theta)\\sin(\varphi) sin(\theta)\\cos(\theta)
     \end{pmatrix}$

## **Team Contacts**

For help, questions, and/or conscerns, please reach out to:

- Yunhao Ba (yhba at ucla dot edu)
- Alex Gilbert (alexrgilbert at ucla dot edu)
- Franklin Wang (franklinxzw at gmail dot com)