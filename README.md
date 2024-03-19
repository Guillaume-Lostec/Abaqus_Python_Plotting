# Abaqus_Python_Plotting

<p align="justify">
Motivation
</p>

**1) What it does (and why)**

<p align="justify">
Let me first briefly describe my context. I am simulating a pure shear sample stretched with constant speed. The material is hyperelastic (Neo Hookean) and Mullins effect has been added with the built-in (Ogden-Roxburgh) option in Abaqus. In the middle there is a layer of cohesive elements with traction-separation law to allow the crack to grow. More information about the model in this [publication](https://doi.org/10.1016/j.eml.2022.101726). I am interested in defining a damage measure from the DMENER and SENER field outputs of abaqus. DMENER is the damage dissipation energy density at integration points, let's write it \Psi_d. SENER is the strain energy density at integration points, let's write it \Psi_e. My damage measure will be d=\Psi_d/(\Psi_e+\Psi_d) and I want to show its value as a color over the mesh. It is possible to do this in Abaqus CAE by operating on the field outputs (see documentation). But I am interested in doing it in Python because ultimately I know I will want more freedom in creating my figures for scientific publication. So in the following I will describe how to create the following GIF that shows my custom damage measure d over the mesh for all timesteps of the simulation:
</p>
