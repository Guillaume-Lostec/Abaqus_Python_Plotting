# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

def Macro1():
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=0)
    odb = session.odbs['path_where_your_job_is/Job-1.odb']
    session.writeFieldReport(fileName='Report_Files/0.rpt', append=ON, 
        sortItem='Node Label', odb=odb, step=0, frame=0, outputPosition=NODAL, 
        variable=(('DMENER', INTEGRATION_POINT), ('SENER', INTEGRATION_POINT), 
        ), stepFrame=SPECIFY)


