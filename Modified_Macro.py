# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
for fr in range(0,101):
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=fr)
    odb = session.odbs['Replace_by_relevant_path\Job-1.odb']
    session.writeFieldReport(fileName=str(fr)+'.rpt', append=ON, 
        sortItem='Node Label', odb=odb, step=0, frame=fr, outputPosition=NODAL, 
        variable=(('LE', INTEGRATION_POINT, ((COMPONENT, 'LE11'), (COMPONENT, 
        'LE22'), (COMPONENT, 'LE33'), )), ), stepFrame=SPECIFY)
    session.writeFieldReport(fileName='U_'+str(fr)+'.rpt', append=ON, 
        sortItem='Node Label', odb=odb, step=0, frame=fr, outputPosition=NODAL, 
        variable=(('U', NODAL, ((COMPONENT, 'U1'), (COMPONENT, 'U2'), )), ), 
        stepFrame=SPECIFY)

