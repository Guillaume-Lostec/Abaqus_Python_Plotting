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
    odb = session.odbs['path_where_your_job_is/Job-1.odb']
    session.writeFieldReport(fileName='Report_Files/'+str(fr)+'.rpt', append=ON, 
        sortItem='Node Label', odb=odb, step=0, frame=fr, outputPosition=NODAL, 
        variable=(('DMENER', INTEGRATION_POINT), ('SENER', INTEGRATION_POINT), 
        ), stepFrame=SPECIFY)


