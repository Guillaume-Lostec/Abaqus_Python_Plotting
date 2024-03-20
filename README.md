# Abaqus_Python_Plotting

<p align="justify">
After running a job in Abaqus, it is possible to produce report files in the ODB to export field outputs (at nodes, integration points, ...). These files can then be parsed in Python to recover the data and it is then possible to operate on it and create figures with more freedom than within the CAE. I had to do this for my research and I realized it was more tricky than I initially thought. To save y'all some time, I share my code and some insights about this process.

**1) What it does (and why).**

<p align="justify">
Let me first briefly describe my context. I am simulating a pure shear sample, stretched with constant speed. The material is hyperelastic (Neo Hookean) and Mullins effect has been added with the built-in (Ogden-Roxburgh) option in Abaqus. In the middle there is a layer of cohesive elements with traction-separation law to allow the crack to grow. More information about the model in this publication https://doi.org/10.1016/j.eml.2022.101726. I am interested in defining a damage measure from Abaqus DMENER and SENER field outputs. DMENER is the damage dissipation energy density at integration points, let's write it \Psi_d. SENER is the strain energy density at integration points, let's write it \Psi_e. My damage measure will be d=\Psi_d/(\Psi_e+\Psi_d) and I want to show its value as a color over the mesh. It is possible to do this in Abaqus CAE by operating on the field outputs (see documentation). But I am interested in doing it in Python because ultimately I know I will want more freedom in creating my figures. So this page describes how to create the following GIF that shows my custom damage measure d over the mesh for all timesteps of the simulation:
</p>

![screenshot](https://github.com/Guillaume-Lostec/Abaqus_Python_Plotting/blob/main/animation.gif)

**2) Getting the report files from Abaqus**

**2.1) Getting the report files with nodal values of DMENER and SENER**

<p align="justify">
We will get 1 report file for each time step of the simulation. The report files will contain the nodes labels as the first column and the nodal values of DMENER and SENER as the 2nd and 3rd columns. My simulation has 100 time steps, so I don't want to get my report files manually, I will therefore use abaqus macros to get them. Go to File/Macro Manager and Create Macro in your work directory. Then go to Report/Field Output: in Step/Frames select Step 1, frame 0. In Variables select Position: Unique Nodal and check the DMENER and SENER boxes (note: you do have to request the outputs when you run your job, if not it will not be there, I assume you did). In Setup, name it Report_Files/0.rpt (create a folder Report_files in your work directory beforehand). Sort by Node Label (the rest can be left as default). Then hit ok, stop recording the macro. You should see a file 0.rpt appeared in your Report_Files folder: go ahead and delete it, what we are interested in is the macro (a python file, likely named abaqusMacro.py) that appeared in your work directory. If you followed the instructions, this file should look like mine (see Initial_Macro.py).

  Now let's add a for loop in that file so we can get all time steps. Open Initial_Macro.py file and add a for loop over all frames: the file should now look like Modified_Macro.py. Open the ODB file and go to File/Run Script, select Modified_Macro.py: it will fill the Report_Files folder with files corresponding to each frame.
</p>

**2.2) Getting the nodes coordinates in the reference configuration.**

<p align="justify">
We want to plot d over the mesh in python in the reference configuration (frame 0). Assuming you requested COORDS as an output in your Job.inp file, you now want to export it. In this case I noticed exporting as a report file gives slighlty off coordinates: I think it is because Abaqus treats COORDS as any other field output and there is some averaging operation over elements (that I don't fully understand). Using the query tool though, I know the exported coordinates are slightly off. When exporting as CSV however, this problem doesn't occur, so that is what I did. So do Report/Field Output, then select Step 1, frame 0. In Variable, select Position: Node Label, and check the COORD1 and COORD2 boxes. In Setup, name it Coords.csv and in Output Format select Comma-separated values (CSV). Sort by Node Label, keep other options as default and hit ok.
</p>

**2.3) Getting the element-nodes correspondence.**

<p align="justify">
The last information we need to extract from Abaqus is the correspondence between elements and nodes. For this, we will get a report file for frame 0 and sort by element label instead of node label, we can pick any field output since we do not care about the values. So do Report/Field Output, select step 1 frame 0. In Variables, do Position: Element Nodal and check any box available. In Setup, name it Elements_Nodes.rpt and Sort by: Element Label, keep the rest as default and hit ok. It will create a file akin to the one I provided as Elements_Nodes.rpt: the first column gives the element label and the second column gives the corresponding node labels, this is the information we need. Note also that if your model has different element types, it will separate them in the report file in 'Regions'. In my case I have 3 regions because I have some bulk elements with 4 nodes (Region 1), some with 3 nodes (Region 2) and some cohesive elements (Region 3).
</p>


**3) Parsing the report files**

<p align="justify">
I provided the code I use to create the gif as Parse_Reports.py. In this code I define 3 functions to parse the information from each of the report file types we just created. The function parse_structure uses Elements_Nodes.rpt to create Elements_Tri and Elements_Quad that contain 3 and 4 colums respectively and give the node labels of each elements: they will be important to create a Triangulation later and plot the mesh. The function parse_coordinates simply extracts X and Y, the coordinates of the nodes in the reference configuration, from Coords.csv. The function parse_fields_values gets the values of DMENER and SENER at nodes from all report files frame.rpt. Since I did not know a simple way to make the plot I had in mind with a mixture of triangular and quad elements, I decided to cut each quad element into 2 triangles instead. The function that does that is called Quad2Tri, it starts by creating a convex hull for each quad element (because Elements_Nodes.rpt gives us the list of nodes for each element but doesn't tell us how they are connected). By crating a convex hull, it is then possible to cut each quad element into adjascent triangles (if we didn't do that, the 2 triangles would be formed between random nodes of the elements and would most likely intersect rather than divide the element properly). More details are given as comments in the file. Once quad elements have been cut, we are left with only trinagular elements: we may then create a triangulation and use it into tricontourf to plot our field d=\Psi_d/(\Psi_e+\Psi_d) as a color over the mesh. We can repeat this for all frames and create the animated gif shown above.
</p>


<p align="justify">
It is unlikely you are trying to do the exact same post-processing, but I am hoping this example will save you some time with your problem.
</p>
