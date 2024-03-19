# Abaqus_Python_Plotting

<p align="justify">
After running a job in Abaqus, it is possible to produce report files in the ODB to export field outputs (at nodes, integration points, ...). These files can then be parsed in Python to recover the data and it is then possible to operate on it and create figures with more freedom than within the CAE. I had to do this for my research and I realized it was more tricky than I initially thought. To save y'all some time, I share my code and some insights about this process.

**1) What it does (and why).**

<p align="justify">
Let me first briefly describe my context. I am simulating a pure shear sample, stretched with constant speed. The material is hyperelastic (Neo Hookean) and Mullins effect has been added with the built-in (Ogden-Roxburgh) option in Abaqus. In the middle there is a layer of cohesive elements with traction-separation law to allow the crack to grow. More information about the model in this publication https://doi.org/10.1016/j.eml.2022.101726. I am interested in defining a damage measure from Abaqus DMENER and SENER field outputs. DMENER is the damage dissipation energy density at integration points, let's write it \Psi_d. SENER is the strain energy density at integration points, let's write it \Psi_e. My damage measure will be d=\Psi_d/(\Psi_e+\Psi_d) and I want to show its value as a color over the mesh. It is possible to do this in Abaqus CAE by operating on the field outputs (see documentation). But I am interested in doing it in Python because ultimately I know I will want more freedom in creating my figures for scientific publication. So this page describes how to create the following GIF that shows my custom damage measure d over the mesh for all timesteps of the simulation:
</p>

![screenshot](https://github.com/Guillaume-Lostec/Abaqus_Python_Plotting/blob/main/animation.gif)

**2) Getting the report files from Abaqus**

**2.1) Getting the report files with nodal values of DMENER and SENER**

<p align="justify">
We will get 1 report file for each time step of the simulation. The report files will contain the nodes labels as the first column and the nodal values of DMENER and SENER as the 2nd and 3rd columns. My simulation has 100 time steps, so I don't want to get my report files manually, I will therefore use abaqus macros to get them. Go to File/Macro Manager and Create Macro in your work directory. Then go to Report/Field Output: in Step/Frames select Step 1, frame 0. In Variables select Position: Unique Nodal and check the DMENER and SENER boxes (note: you do have to request the outputs when you run your job, if not it will not be there, I assume you did). In Setup, name it Report_Files/0.rpt (create a folder Report_files in your work directory beforehand). Sort by Node Label (the rest can be left as default). Then hit ok, stop recording the macro. You should see a file 0.rpt appeared in your Report_Files folder: go ahead and delete it, what we are interested in is the macro (a python file, likely named abaqusMacro.py) that appeared in your work directory. If you followed the instructions, this file should look like mine (see Initial_Macro.py).

  Now let's add a for loop in that file so we can get all time steps.
</p>

**3) Parsing the report files**
