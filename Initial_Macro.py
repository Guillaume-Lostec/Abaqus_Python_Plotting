# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0)
odb = session.odbs['Replace_by_relevant_path\Job-1.odb']
session.writeFieldReport(fileName='0.rpt', append=ON, 
    sortItem='Node Label', odb=odb, step=0, frame=0, outputPosition=NODAL, 
    variable=(('LE', INTEGRATION_POINT, ((COMPONENT, 'LE11'), (COMPONENT, 
    'LE22'), (COMPONENT, 'LE33'), )), ), stepFrame=SPECIFY)
session.writeFieldReport(fileName='U_0.rpt', append=ON, 
    sortItem='Node Label', odb=odb, step=0, frame=0, outputPosition=NODAL, 
    variable=(('U', NODAL, ((COMPONENT, 'U1'), (COMPONENT, 'U2'), )), ), 
    stepFrame=SPECIFY)
