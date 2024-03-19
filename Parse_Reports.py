import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import csv
from scipy.spatial import ConvexHull
from PIL import Image
import os

def parse_structure(correspondence,start_quad,stop_quad,start_tri,stop_tri):
    # Note we don't record Abaqus labels for elements. 
    # They may not be the same as the index in the tables we are creating (Elements_Quad and Elments_Tri).
    # It's not a problem for plotting in python. 
    # However be careful when comparing elements between these tables and the CAE: they might have different labels.
    try:
        # Read the file and build Elements_Tri (1st region)
        nodes_quad=[]
        with open(correspondence, 'r') as f:
            lines = f.readlines()
        for i in range(start_quad,stop_quad):
            values = lines[i].split()
            nodes_quad.append(float(values[1]))
    except FileNotFoundError:
        pass
    try:
        # Read the file and build Elements_Tri (2nd region)
        nodes_tri=[]
        with open(correspondence,'r') as f:
            lines = f.readlines()
        for i in range(start_tri,stop_tri):
            values = lines[i].split()
            nodes_tri.append(float(values[1]))
    except FileNotFoundError:
        pass
    nodes_quad=np.array(nodes_quad)
    nodes_tri=np.array(nodes_tri)
    N4=int(np.size(nodes_quad)/4)
    N3=int(np.size(nodes_tri)/3)
    Elements_Quad=nodes_quad.reshape(N4,4)
    Elements_Tri=nodes_tri.reshape(N3,3)
    Elements_Quad.astype(int)
    Elements_Tri.astype(int)
    return Elements_Quad,Elements_Tri

def parse_coordinates(from_report,coordinates,start_coords,stop_coords,stop_csv):
    X=[]
    Y=[]
    if from_report:
        try:
            # Read the file and get nodes coordinates
            with open(coordinates, 'r') as f:
                lines = f.readlines()
            for i in range(start_coords,stop_coords):
                values = lines[i].split()
                X.append(float(values[1]))
                Y.append(float(values[2]))
        except FileNotFoundError:
            pass
        X=np.array(X)
        Y=np.array(Y)
    else: # from CSV
        with open(coordinates,'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header
            for i,row in enumerate(csvreader):
                if i>=stop_csv:
                    break
                X.append(float(row[5]))
                Y.append(float(row[6]))
        X=np.array(X)
        Y=np.array(Y)
        X.astype(float)
        Y.astype(float)
    return X,Y

def Quad2Tri(Elements_Quad,X,Y):
# There is no specific logic in Abaqus' numbering of quadric elements' nodes.
# I want to enforce that the 2 resulting triangles share a common diagonal from the original quadratic element (and not 2 intersecting ones).
# I start by creating the vertices of the quad element: using ConvexHull.
# For node Elements_Quad[i][0], I use hull.simplices to detect which 2 nodes it's connected to within the Quad element (Connections).
# I make a first triangle w nodes Elements_Quad[i][0] and the 2 nodes in Connections.
# The other triangle should be formed by the remaining node in Elements_Quad[i] and the 2 nodes of Connections.
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

def parse_fields_values(start_fields,stop_fields):
    DMENER=[]
    SENER=[]
    for file_number in range(0,101):
        file=f"Report_Files/{file_number}.rpt"
        try:
            # Read the file and get values of field(s) at nodes
            dmener=[]
            sener=[]
            with open(file,'r') as f:
                lines = f.readlines()
            for i in range(start_fields,stop_fields):
                values = lines[i].split()
                dmener.append(float(values[1]))
                sener.append(float(values[2]))
        except FileNotFoundError:
            pass
        DMENER.append(dmener)
        SENER.append(sener)
    DMENER=np.array(DMENER)
    SENER=np.array(SENER)
    DMENER=np.transpose(DMENER)
    SENER=np.transpose(SENER)
    return DMENER,SENER

## Get Elements-Nodes correspondence tables
correspondence=f"Report_Files/Elements_Nodes.rpt"
start_quad=22 # Manually enter those 'how-to-read instructions' (look through the file and modify accordingly)
stop_quad=10614
start_tri=10635
stop_tri=10803
Elements_Quad,Elements_Tri=parse_structure(correspondence,start_quad,stop_quad,start_tri,stop_tri)

## Get nodes coordinates in reference frame
coordinates=f"Report_Files/Coords.csv"
from_report=0
start_coords=22 # Manually enter those 'how-to-read instructions' (look through the file and modify accordingly)
stop_coords=2900
stop_csv=2878
X,Y=parse_coordinates(from_report,coordinates,start_coords,stop_coords,stop_csv)

## Transform each Quadratric Elt into 2 Triangular elts
Now_Tri=Quad2Tri(Elements_Quad,X,Y)
Elements=np.concatenate((Now_Tri,Elements_Tri),axis=0)
Elements=Elements-np.ones_like(Elements) # bc Abaqus numbers nodes from 1 to N while Python starts indices at 0: so for triangulation it needs to be adjusted
Now_Tri=Now_Tri-np.ones_like(Now_Tri)
Elements_Tri=Elements_Tri-np.ones_like(Elements_Tri)

## Get values of fields at each time steps from report files
start_fields=22 # Manually enter those 'how-to-read instructions' (look through the file and modify accordingly)
stop_fields=2900
DMENER,SENER=parse_fields_values(start_fields,stop_fields)

## Plot field over the mesh (with now only triangular elements)
xmax=400
xmin=0
ymax=25
ymin=-25
fig,ax=plt.subplots(figsize=((xmax-xmin),(ymax-ymin)))
triang=Triangulation(X,Y,Elements)
Damage=DMENER[:,50]/(DMENER[:,50]+SENER[:,50])
plt.tricontourf(triang,Damage,cmap='jet')
plt.triplot(triang,color='k',linewidth=1.0)
threshold=0.3
contour=plt.tricontour(triang,Damage,levels=[threshold],colors='black',linewidths=40.0,zorder=1)

# ## Make a gif to show all time steps
# frames=[]
# image_files=[]
# xmax=40
# xmin=0
# ymax=2.5
# ymin=-2.5
# threshold=0.8
# triang=Triangulation(X,Y,Elements)
# for i in range(1,101):
#     Damage=DMENER[:,i]/(DMENER[:,i]+SENER[:,i])
#     fig,ax=plt.subplots(figsize=((xmax-xmin),(ymax-ymin)))
#     plt.tricontourf(triang,Damage,cmap='jet',vmin=0.0,vmax=1.0)
#     plt.triplot(triang,color='k',linewidth=1.0)
#     contour=plt.tricontour(triang,Damage,levels=[threshold],colors='white',linewidths=5.0,zorder=1)
#     plt.tight_layout()
#     filename = f"frame_{i:03d}.png"
#     plt.savefig(filename)
#     plt.close()
#     frames.append(Image.open(filename))
#     image_files.append(filename)
# frames[0].save("Damage.gif", save_all=True, append_images=frames[1:],duration=100,loop=0)
# for filename in image_files:
#     os.remove(filename)