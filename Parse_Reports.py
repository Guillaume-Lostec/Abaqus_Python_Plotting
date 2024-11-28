import numpy as np
import re
from collections import defaultdict
import csv
from scipy.spatial import ConvexHull
from scipy import special
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.optimize import curve_fit
from PIL import Image
import os
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def parse_structure(correspondence):
    # In this function, 'regions' correspond to Abaqus' language (see documentation)
    # For this case: regio 1 = 4 nodes elements, region 2 = 3 nodes elements and region 3 = cohesive elements.
    # Note we don't record Abaqus labels for elements. 
    # They may not be the same as the index in the tables we are creating (Elements_Quad and Elments_Tri).
    # It's not a problem for plotting in python. 
    # However be careful when comparing elements between these tables and the CAE: they might have different labels.
    regions_data = defaultdict(dict)
    current_region = None
    region_pattern = re.compile(r'Field Output reported at element nodes for region: PART-\d+-\d+\.Region_(\d+)')
    element_node_pattern = re.compile(r'\s*(\d+)\s+(\d+)\s+[\d.]+') 
    with open(correspondence, 'r') as file:
        for line in file:
            # Check if starting a new region
            region_match = region_pattern.search(line)
            if region_match:
                current_region = int(region_match.group(1))
                continue
            # Check if reading element/node data
            if current_region:
                element_node_match = element_node_pattern.match(line)
                if element_node_match:
                    element_label = int(element_node_match.group(1))
                    node_label = int(element_node_match.group(2))
                    # Append node label to the element in the current region
                    if element_label not in regions_data[current_region]:
                        regions_data[current_region][element_label] = []
                    regions_data[current_region][element_label].append(node_label)
    parsed_data=dict(regions_data)
    region_1_data = parsed_data[1]
    region_2_data = parsed_data[2]
    Elements_Quad = np.array([region_1_data[element] for element in region_1_data])
    Elements_Tri = np.array([region_2_data[element] for element in region_2_data])
    Elements_Quad.astype(int)
    Elements_Tri.astype(int)
    return Elements_Quad,Elements_Tri

def parse_coordinates(coordinates,start_fields_r1,stop_fields_r1,Missing_nodes,Line_numbers):
    # Get coordinates of most nodes
    X=[]
    Y=[]
    stop_csv=stop_fields_r1-start_fields_r1
    with open(coordinates,'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for i,row in enumerate(csvreader):
            if i>=stop_csv:
                break
            X.append(float(row[5]))
            Y.append(float(row[6]))
    # Insert coordinates of 'missing nodes' (see more explanations in the function parse_field_values)
    Corresp_line_numbers=Line_numbers-start_fields_r1*np.ones_like(Line_numbers)-19*np.ones_like(Line_numbers) # 19 corresponds to the lines in the fields report files in between region 1 and 2
    with open(coordinates,'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        k=0
        for i,row in enumerate(csvreader):
            if i in Corresp_line_numbers:
                X.insert(Missing_nodes[k]-1,float(row[5]))
                Y.insert(Missing_nodes[k]-1,float(row[6]))
                k=k+1
    X=np.array(X)
    Y=np.array(Y)
    X.astype(float)
    Y.astype(float)
    return X,Y

def Quad2Tri(Elements_Quad,X,Y):
# There is no specific logic in the numbering of quadric elements' nodes
# I want to enforce that the 2 resulting triangles share a common diagonal from the original quadratic element (and not 2 intersecting ones)
# I start by creating the vertices of the quad element: using ConvexHull
# For node Elements_Quad[i][0], I use hull.simplices to detect which 2 nodes (call it A,B) it's connected to within the Quad element
# I make a first triangle w nodes Elements_Quad[i][0], A and B
# The other triangle should be formed by the remaining node in Elements_Quad[i], A and B.
    Now_Tri = np.zeros([2*np.size(Elements_Quad,0),3])
    for i in range(np.size(Elements_Quad,0)):
        # Create hull
        Coords_Quad=[[X[int(Elements_Quad[i][0])-1],Y[int(Elements_Quad[i][0])-1]],\
                    [X[int(Elements_Quad[i][1])-1],Y[int(Elements_Quad[i][1])-1]],\
                    [X[int(Elements_Quad[i][2])-1],Y[int(Elements_Quad[i][2])-1]],\
                    [X[int(Elements_Quad[i][3])-1],Y[int(Elements_Quad[i][3])-1]]]
        Coords_Quad=np.array(Coords_Quad)
        Hull=ConvexHull(Coords_Quad)
        # Find nodes connected to Elements_Quad[i][0]
        Connections=[]
        for v in range(0,np.size(Hull.simplices,0)):
            vertex=Hull.simplices[v]
            if 0 in vertex:
                other_one = vertex[0] if vertex[0] !=0 else vertex[1]
                Connections.append(int(Elements_Quad[i][other_one]))
        # First triangle
        Now_Tri[2*i][0] = int(Elements_Quad[i][0])
        Now_Tri[2*i][1] = Connections[0]
        Now_Tri[2*i][2] = Connections[1]
        # 2nd triangle
        Now_Tri[2*i+1][0] = Connections[0]
        Now_Tri[2*i+1][1] = Connections[1]
        Remaining=[int(Elements_Quad[i][1]),int(Elements_Quad[i][2]),int(Elements_Quad[i][3])]
        Remaining.remove(Connections[0])
        Remaining.remove(Connections[1])
        Now_Tri[2*i+1][2] = Remaining[0]
    return Now_Tri

def parse_fields_values(time_steps):
    # In this function, we want to extract the values of fields at the nodes.
    # The nodal values are given by Abaqus in groups belonging to the same region.
    # Therefore the value of fields at node N can be reported in multiple regions.
    # Typically, if a node N is shared between a quadrangular and a triangular element, it will be reported in bothe region 1 and region 2.
    # Most nodes that belong to triangular elements also belong to quadrangular ones, therefore it seems like parsing only region 1 could be sufficient.
    # However, there exist a few nodes of triangular elements that are not also connected to quadrangualr elements (typically in the corners).
    # Those are what I call in the code 'missing nodes'.

    # Detect 'missing nodes'
    file_path=f"0.rpt"
    region_data = defaultdict(list)
    current_region = None
    region_pattern = re.compile(r'Field Output reported at nodes for region: PART-\d+-\d+\.Region_(\d+)')
    node_data_pattern = re.compile(r'\s*(\d+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)')
    with open(file_path, 'r') as file:
        for line in file:
            # Check for region
            region_match = region_pattern.search(line)
            if region_match:
                current_region = int(region_match.group(1))
                continue
            # Check for node data
            if current_region:
                node_data_match = node_data_pattern.match(line)
                if node_data_match:
                    node_label = int(node_data_match.group(1))
                    region_data[current_region].append(node_label)
    parsed_data=dict(region_data)
    region_1_data = parsed_data[1]
    start_fields_r1=22 # it is always
    stop_fields_r1=start_fields_r1+len(region_1_data)
    Nodes_labels=np.array(region_1_data)
    Missing_nodes = []
    for i in range(1, len(Nodes_labels)):
        if Nodes_labels[i] - Nodes_labels[i-1] > 1:
            for j in range(int(Nodes_labels[i-1])+1, int(Nodes_labels[i])):
                Missing_nodes.append(j)
    # Detect which line in the region 2 (triangular elts) of the file corresponds to those missing nodes
    Line_numbers=[]
    region_2_data = parsed_data[2]
    for i in range(0,len(region_2_data)):
        if int(region_2_data[i]) in Missing_nodes:
            Line_numbers.append(i)
    # Get all values for nodes belonging to quadrangular elts
    LE11=[]
    LE22=[]
    LE33=[]
    for file_number in range(0,time_steps+1):
        file_path=f"{file_number}.rpt"
        region_data = defaultdict(list)
        current_region = None
        region_pattern = re.compile(r'Field Output reported at nodes for region: PART-\d+-\d+\.Region_(\d+)')
        node_data_pattern = re.compile(r'\s*(\d+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)')
        with open(file_path, 'r') as file:
            for line in file:
                # Check for region
                region_match = region_pattern.search(line)
                if region_match:
                    current_region = int(region_match.group(1))
                    continue
                # Check for node data
                if current_region:
                    node_data_match = node_data_pattern.match(line)
                    if node_data_match:
                        le11 = float(node_data_match.group(2)) 
                        le22 = float(node_data_match.group(3)) 
                        le33 = float(node_data_match.group(4)) 
                        region_data[current_region].append([le11, le22, le33])
        parsed_data=dict(region_data)
        region_1_data = parsed_data[1]
        region_2_data = parsed_data[2]
        le123=np.array(region_1_data)
        le11=le123[:,0]
        le22=le123[:,1]
        le33=le123[:,2]
        # Insert the missing nodal values
        for mssg in range(len(Missing_nodes)):
                values = region_2_data[Line_numbers[mssg]]
                le11.insert(Missing_nodes[mssg]-1,float(values[0]))
                le22.insert(Missing_nodes[mssg]-1,float(values[1]))
                le33.insert(Missing_nodes[mssg]-1,float(values[2]))
        LE11.append(le11)
        LE22.append(le22)
        LE33.append(le33)
    LE11=np.transpose(LE11)
    LE22=np.transpose(LE22)
    LE33=np.transpose(LE33)
    return LE11,LE22,LE33,Missing_nodes,Line_numbers,start_fields_r1,stop_fields_r1

def parse_crack_position(stress_coh,correspondence,start_s22,stop_s22,X,time_steps):
    # Get the first non-zero stress element
    out=[]
    try:
        with open(stress_coh, 'r') as f:
            lines = f.readlines()
        for i in range(start_s22,stop_s22):
            values = lines[i].split()
            out.append(values)
    except FileNotFoundError:
        pass
    out=np.array(out)
    out2=out.astype(float)
    out3=out2[:,1:]
    first_nonzero=np.zeros_like(out3[:,0])
    for t in range(1,time_steps+1):
        leave=0
        for el in range(0,len(out3[0,:])):
            if out3[t,el]>1e-20 and leave==0:
                leave=1
                first_nonzero[t]=el
            if el==len(out3[0,:])-1 and leave==0:
                first_nonzero[t]=el
    # Get the element node labels
    regions_data = defaultdict(dict)
    current_region = None
    region_pattern = re.compile(r'Field Output reported at element nodes for region: PART-\d+-\d+\.Region_(\d+)')
    element_node_pattern = re.compile(r'\s*(\d+)\s+(\d+)\s+[\d.]+') 
    with open(correspondence, 'r') as file:
        for line in file:
            # Check if starting a new region
            region_match = region_pattern.search(line)
            if region_match:
                current_region = int(region_match.group(1))
                continue
            # Check if reading element/node data
            if current_region:
                element_node_match = element_node_pattern.match(line)
                if element_node_match:
                    element_label = int(element_node_match.group(1))
                    node_label = int(element_node_match.group(2))
                    # Append node label to the element in the current region
                    if element_label not in regions_data[current_region]:
                        regions_data[current_region][element_label] = []
                    regions_data[current_region][element_label].append(node_label)
    parsed_data=dict(regions_data)
    region_3_data = parsed_data[3]
    Elements_Quad = np.array([region_3_data[element] for element in region_3_data])
    Elements_Quad=Elements_Quad-np.ones_like(Elements_Quad)
    Elements_Quad.astype(int)
    # Use X and Element_Quad to get c0 and c
    c0=(X[int(Elements_Quad[0,0])]+X[int(Elements_Quad[0,1])]+X[int(Elements_Quad[0,2])]+X[int(Elements_Quad[0,3])])/4
    c=np.zeros(time_steps+1)
    c[0]=c0
    for t in range(1,time_steps+1):
        el=int(first_nonzero[t])
        if el>0:
            c[t]=(X[int(Elements_Quad[el,0])]+X[int(Elements_Quad[el,1])]+X[int(Elements_Quad[el,2])]+X[int(Elements_Quad[el,3])])/4
        else:
            c[t]=c0
    return c0,c,Elements_Quad,out3

def parse_motion(time_steps,sizeX0):
    U1=[]
    U2=[]
    for file_number in range(0,time_steps+1):
        file=f"U_{file_number}.rpt"
        try:
            # Read the file and get values of field(s) at nodes
            u1=[]
            u2=[]
            with open(file,'r') as f:
                lines = f.readlines()
            for i in range(19,19+sizeX0):
                values = lines[i].split()
                u1.append(float(values[1]))
                u2.append(float(values[2]))
        except FileNotFoundError:
            pass
        U1.append(u1)
        U2.append(u2)
    U1=np.array(U1)
    U2=np.array(U2)
    U1=np.transpose(U1)
    U2=np.transpose(U2)
    return U1,U2

dt=1.0
Mullins=[2.0,0.5,0.5]
time_steps=100
stop_time=int(time_steps*dt)
## Get Elements-Nodes correspondence tables
correspondence=f"Elements_Nodes.rpt"
Elements_Quad,Elements_Tri=parse_structure(correspondence)
## Get values of fields at each time steps from report files
LE11,LE22,LE33,Missing_nodes,Line_numbers,start_fields_r1,stop_fields_r1=parse_fields_values(time_steps)
## Get nodes coordinates in reference frame
coordinates=f"Coords.csv"
X0,Y0=parse_coordinates(coordinates,start_fields_r1,stop_fields_r1,Missing_nodes,Line_numbers)
## Get nodes displacement at each time steps from report files
U1,U2=parse_motion(time_steps,np.size(X0))
X=np.zeros([np.size(X0),time_steps+1])
Y=np.zeros([np.size(Y0),time_steps+1])
X[:,0]=X0
Y[:,0]=Y0
for tt in range(1,time_steps+1):
    X[:,tt]=X0+U1[:,tt]
    Y[:,tt]=Y0+U2[:,tt]
## Transform each Quadratric Elt into 2 Triangular elts
Now_Tri=Quad2Tri(Elements_Quad,X0,Y0)
Elements=np.concatenate((Now_Tri,Elements_Tri),axis=0)
Elements=Elements-np.ones_like(Elements) # bc Abaqus numbers nodes from 1 to N while Python starts indices at 0: so for triangulation it needs to be adjusted
Now_Tri=Now_Tri-np.ones_like(Now_Tri)
Elements_Tri=Elements_Tri-np.ones_like(Elements_Tri)
## Get crack position from cohesive stress
stress_coh=f"Coh.rpt"
start_s22=5
stop_s22=time_steps+start_s22+1
correspondence=f"Elements_Nodes.rpt"
c0,c,Elements_Coh,Stress_Coh=parse_crack_position(stress_coh,correspondence,start_s22,stop_s22,X0,time_steps)

## Get strain energy density from LE output
TrC=np.exp(2*LE11)+np.exp(2*LE22)+np.exp(2*LE33) # Trace of Right Cauchy Green tensor C
detC=np.exp(2*LE11)*np.exp(2*LE22)*np.exp(2*LE33) # determinant of C
J=np.sqrt(detC) # Jacobian
C10=5.0
D10=10000.0
W=C10*(J**(-2/3)*TrC-3*np.ones_like(TrC))+D10*(J-np.ones_like(J))**2 # Nominal strain energy density
eta=np.zeros_like(W) # damage variable from Ogden-Roxburgh model
r=Mullins[0]
m=Mullins[1]
beta=Mullins[2]
for n in range(0,np.size(W,0)): # nodes
    for t in range(0,np.size(W,1)): # times
        if t>0:
            Wmax=np.max(W[n,0:t+1])
        else:
            Wmax=np.max(W[n,0])
        eta[n,t]=1-1/r*special.erf((Wmax-W[n,t])/(m+beta*Wmax))
Phi=(np.ones_like(W)-eta)*W # damage strain energy density
Psi=eta*W # strain energy density

## Plot the strain energy density field at time step 50, on undeformed mesh
xmax=500
xmin=0
ymax=50
ymin=-50
fig,ax=plt.subplots(figsize=((xmax-xmin)/10,(ymax-ymin)/10))
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
triang=Triangulation(X0,Y0,Elements)
ax.tricontourf(triang,Psi[:,50],cmap='inferno')
plt.triplot(triang,color='w',linewidth=0.2)
ax.plot([0,c0],[0,0],'w',linewidth=4.0)
ax.scatter(c0,0,40,'w')
plt.show()
fig.savefig('SED_undeformed.png')

## Gif Psi(t)
frames=[]
image_files=[]
for i in range(0,101):
    xmin, xmax = np.min(X[:,i])/10, np.max(X[:,i])/10
    ymin, ymax = np.min(Y[:,i])/10, np.max(Y[:,i])/10
    fig,ax=plt.subplots(figsize=((xmax-xmin),(ymax-ymin)))
    triang = Triangulation(X[:,i], Y[:,i], Elements)
    ax.tricontourf(triang,Psi[:, i], cmap='inferno')
    filename = f"frame_{i:03d}.png"
    plt.axis('off')
    plt.savefig(filename)
    plt.close()
    frames.append(Image.open(filename))
    image_files.append(filename)
output_size = (5000,1978)
centered_frames = []
for frame in frames:
    padded_frame = Image.new("RGBA", output_size, (255, 255, 255, 255))
    offset = ((output_size[0] - frame.width) // 2, (output_size[1] - frame.height) // 2)
    padded_frame.paste(frame, offset)
    centered_frames.append(padded_frame)
centered_frames[0].save("Animation_Psi.gif", save_all=True, append_images=centered_frames[1:], duration=50, loop=0)
for filename in image_files:
    os.remove(filename)

## Resistance curve G(c)
Time=np.linspace(0,stop_time,time_steps+1)
du_dt=50.0/100.0
u_top=np.linspace(0,du_dt*stop_time,time_steps+1)
H0=50.0
# _ff is for 'far field', with stretch state:
lambda2=1+u_top/H0
lambda1=np.ones_like(lambda2)
lambda3=1/lambda2
TrC_ff=lambda1**2+lambda2**2+lambda3**2
W_ff=C10*(TrC_ff-3*np.ones_like(TrC_ff))
G=2*H0*W_ff
fig,ax=plt.subplots(figsize=(7,5))
Delta=4.0
Gamma0=200.0
ax.plot((c-c0)/Delta,np.ones_like(G),'r')
ax.scatter((c-c0)/Delta,G/Gamma0,5,'k')
fig.savefig('Resistance_curve.png')