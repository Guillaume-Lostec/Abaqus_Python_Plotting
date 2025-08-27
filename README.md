# Abaqus_Python_Plotting

<p align="justify">
After running a job in Abaqus, it is possible to produce report files in the ODB to export field outputs (at nodes, integration points, ...). These files can then be parsed in Python to recover the data (to do postprocessing and make figures with more freedom than within the CAE). To save y'all some time, I share my code and some insights about this process.

**1) What it does**

<p align="justify">
Let me first briefly describe my context. I am simulating a pure shear sample, stretched with constant rate. The material is hyperelastic (Neo Hookean) and Mullins effect has been added with the built-in (Ogden-Roxburgh) option in Abaqus. In the middle there is a layer of cohesive elements with traction-separation law to allow the crack to grow. More information about the model in this publication https://doi.org/10.1007/s10704-025-00884-w. I ran the simulation with the explicit solver and got an ODB file. Now, I want to:
  
- Make a plot to show the strain energy density psi(t*) (at a given time t*) as a color over the undeformed mesh.

![screenshot](https://github.com/Guillaume-Lostec/Abaqus_Python_Plotting/blob/main/SED_undeformed.png)
  
- Make a gif to show the time evolution of the strain energy density psi(t) as a color over the deformed mesh, as the crack opens and propagates.

![screenshot](https://github.com/Guillaume-Lostec/Abaqus_Python_Plotting/blob/main/Animation_Psi.gif)

- Make a plot G(c) where G(t) is a given function of time c(t) is the crack length.

![screenshot](https://github.com/Guillaume-Lostec/Abaqus_Python_Plotting/blob/main/Resistance_curve.png)
  
Note it is possible to do this in Abaqus CAE by operating on the field outputs (see documentation). But I am interested in doing it in Python because ultimately I know I will want more freedom in creating my figures.
</p>

**2) Getting the report files from Abaqus**

**2.1) Getting the report files with nodal values of the logarithmic strain output (LE) and the displacement of the nodes.**

<p align="justify">
For the logarithmic strain, we will get 1 report file for each time step of the simulation. The report files will contain the nodes labels as the first column and the nodal values of the logarithmic strain in the x-direction (LE.LE11), y-direction (LE.LE22) and z-direction (LE.LE33) as the 2nd to 4th columns. My simulation has 100 time steps, so I don't want to get my report files manually, I will therefore use abaqus macros to get them. Go to File/Macro Manager and Create Macro in your work directory. Then go to Report/Field Output: in Step/Frames select Step 1, frame 0. In Variables select Position: Unique Nodal and check the LE11, LE22 and LE33 boxes (note: you do have to request the LE output in the inp file when you run your job, if not it will not be there, I assume you did). In Setup, name it 0.rpt and sort by Node Label (the rest can be left as default), then hit ok.
For the displacement of the nodes, we also create a file per time step in a similar manner. Go to Report/Field Output: in Step/Frames select Step 1, frame 0. In Variables select Position: Unique Nodal and check the U1 and U2 boxes (again, if these outputs are not available, it's probably because you didn't request them in the inp file). In Setup, name it U_0.rpt and sort by Node Label (the rest can be left as default), hit ok.
Now, stop recording the macro. You should see a file 0.rpt and another U_0.rpt appeared in your folder: go ahead and delete them, what we are interested in is the macro (a python file, likely named abaqusMacro.py) that appeared in your work directory. If you followed the instructions, this file should look like mine (see Initial_Macro.py). Now let's add for loops in that file so we can get all time steps. Open Initial_Macro.py file and add for loops over all frames: the file should now look like Modified_Macro.py. 
Now open the ODB file and go to File/Run Script, select Modified_Macro.py: it will create all the files 0.rpt to 100.rpt and U_0.rpt to U_100.rpt (similar to those I have attached in this repository).
</p>

**2.2) Getting the nodes coordinates in the reference configuration.**

<p align="justify">
We want to plot fields over the mesh in python in the reference configuration (frame 0). Assuming you requested COORDS as an output in your Job.inp file, you now want to export it. In this case I noticed exporting as a report file gives slighlty off coordinates: I am guessing it's because Abaqus treats COORDS as any other field output and there is some averaging operation over elements (that I don't fully understand). Using the query tool though, I know the exported coordinates are slightly off. When exporting as CSV however, this problem doesn't occur, so that is what I did. So do Report/Field Output, then select Step 1, frame 0. In Variable, select Position: Node Label, and check the COORD1 and COORD2 boxes. In Setup, name it Coords.csv and in Output Format select Comma-separated values (CSV). Sort by Node Label, keep other options as default and hit ok.
</p>

**2.3) Getting the element-nodes correspondence.**

<p align="justify">
Another important information we need to extract from Abaqus is the correspondence between elements and nodes. For this, we will get a report file for frame 0 and sort by element label instead of node label, we can pick any field output since we do not care about the values. So do Report/Field Output, select step 1 frame 0. In Variables, do Position: Element Nodal and check any box available. In Setup, name it Elements_Nodes.rpt and Sort by: Element Label, keep the rest as default and hit ok. It will create a file akin to the one I provided as Elements_Nodes.rpt: the first column gives the element label and the second column gives the corresponding node labels, this is the information we need. Note also that if your model has different element types, it will separate them in the report file in 'Regions'. In my case I have 3 regions because I have some bulk elements with 4 nodes (Region 1), some with 3 nodes (Region 2) and some cohesive elements (Region 3).
</p>

**2.4) Getting the stress in the y-direction for cohesive elements.**

<p align="justify">
In my work, I needed to extract the size of the crack, which I define as the X coordinate of the centroid of the first cohesive element for which the Cauchy stress is non-zero, starting from X = c0. This is specific to my problem, but since it requires to create XY Data and export it, it can be useful to go over this. So, open the ODB file and go to 'Create XY Data', select 'ODB Field output', continue: select Position:Centroid, select the S22 box. Then in Element/Nodes, select the set of cohesive elements and hit save. Now you can export a report file for this data in Report/XY Data, select all the lines and export it as Coh.rpt in the Setup menu (I provided the file it creates in the repository).
</p>

**3) Parsing the report files.**

<p align="justify">
I provided the code I use to create the gif as Parse_Reports.py. In this code I define functions to parse the information from each of the report file types we just created. The function parse_structure uses Elements_Nodes.rpt to create Elements_Tri and Elements_Quad that contain 3 and 4 colums respectively and give the node labels of each elements: they will be important to create a Triangulation later and plot the mesh. The function parse_coordinates simply extracts X0 and Y0, the coordinates of the nodes in the reference configuration, from Coords.csv. The function parse_fields_values gets the values of LE11, LE22 and LE33 at nodes from all report files frame.rpt. The function parse_motion gets the displacements in x and y directions of the nodes at all times (this is needed if we wish to plot over the deformed mesh). Finally, a function parse_crack_position is reading through the file Coh.rpt and returns the crack position.
  You will notice those parsing functions use the module 'Regular Expression operations': the idea is to detect patterns in the report files so we can detect where the relevant data is located. It's so we don't have to provide line numbers by hand. Sometimes (like in the parse_motion function), I am reading line by line instead, since we may predict the line numbers from the number of nodes without having to open the file beforehand.
</p>

**3) Plotting.**

<p align="justify">
There is also another function called Quad2Tri: since I did not know a simple way to plot fields over the mesh in python with a mixture of triangular and quad elements, I decided to cut each quad element into 2 triangles instead, this is what Quad2Tri does. Specifically, it starts by creating a convex hull for each quad element (because Elements_Nodes.rpt gives us the list of nodes for each element but doesn't tell us how they are connected). By creating a convex hull, it is then possible to cut each quad element into adjascent triangles (if we didn't do that, the 2 triangles would be formed between random nodes of the elements and would most likely intersect rather than divide the element properly). More details are given as comments in the file. Once quad elements have been cut, we are left with only trinagular elements: we may then create a triangulation and use it into tricontourf to plot fields over the mesh.
  To create the animated gif, note that I am using 'padding' because the frames are stacked on top of each other but are not aligned at the center, so it would create a drift of the figure if you skipped this step.
</p>


<p align="justify">
It is unlikely you are trying to do the exact same post-processing, but I am hoping this example will save you some time with your problem.
</p>
