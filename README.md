# Abaqus_Python_Plotting

<p align="justify">
After running a job in Abaqus, it is possible to produce report files in the ODB to export field outputs (at nodes, integration points, ...). These files can then be parsed in Python to recover the data and it is then possible to operate on it and create figures with more freedom than within the CAE. I had to do this for my research and I realized it was more tricky than I initially thought. To save y'all some time, I share my code and some insights about this process.

**1) What it does (and why).**

<p align="justify">
Let me first briefly describe my context. I am simulating a pure shear sample, stretched with constant speed. The material is hyperelastic (Neo Hookean) and Mullins effect has been added with the built-in (Ogden-Roxburgh) option in Abaqus. In the middle there is a layer of cohesive elements with traction-separation law to allow the crack to grow. More information about the model in this publication https://doi.org/10.1016/j.eml.2022.101726. I am interested in defining a damage measure from Abaqus DMENER and SENER field outputs. DMENER is the damage dissipation energy density at integration points, let's write it \Psi_d. SENER is the strain energy density at integration points, let's write it \Psi_e. My damage measure will be d=\Psi_d/(\Psi_e+\Psi_d) and I want to show its value as a color over the mesh. It is possible to do this in Abaqus CAE by operating on the field outputs (see documentation). But I am interested in doing it in Python because ultimately I know I will want more freedom in creating my figures for scientific publication. So this page describes how to create the following GIF that shows my custom damage measure d over the mesh for all timesteps of the simulation:
</p>

![screenshot](https://github.com/Guillaume-Lostec/Abaqus_Python_Plotting/blob/main/animation.gif)

**2) Getting the report files from Abaqus**

**3) Parsing the report files**
