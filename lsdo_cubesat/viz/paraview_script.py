# state file generated using paraview version 5.8.0-239-g8ba01d39ff

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.8.0-239-g8ba01d39ff
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [2150, 1164]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.39697265625, 0.0, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-39590.68203531288, 14102.11246462003, -7160.963289073443]
renderView1.CameraFocalPoint = [0.3969726562499986, -7.600350449755216e-16, -3.557550822992326e-16]
renderView1.CameraViewUp = [-0.0057356240107452605, 0.4399085138716072, 0.8980242769772327]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 11034.32286362616

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'STL Reader'
sTLReader2 = STLReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/earth_triangulation.stl'])

# create a new 'STL Reader'
sTLReader5 = STLReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/detector_thrust.stl'])

# create a new 'Tecplot Reader'
tecplotReader2 = TecplotReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/detector_position.dat'])

# create a new 'Tecplot Reader'
tecplotReader6 = TecplotReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/sunshade_position_ends.dat'])

# create a new 'Tecplot Reader'
tecplotReader1 = TecplotReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/sunshade_position.dat'])

# create a new 'STL Reader'
sTLReader3 = STLReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/optics_thrust.stl'])

# create a new 'Tecplot Reader'
tecplotReader3 = TecplotReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/optics_position.dat'])

# create a new 'Tecplot Reader'
tecplotReader4 = TecplotReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/detector_position_ends.dat'])

# create a new 'STL Reader'
sTLReader1 = STLReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/earth_sphere.stl'])

# create a new 'STL Reader'
sTLReader4 = STLReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/sunshade_thrust.stl'])

# create a new 'Tecplot Reader'
tecplotReader5 = TecplotReader(FileNames=['/Users/jhwang/OneDrive - UC San Diego/Packages/lsdo_cubesat/lsdo_cubesat/viz/optics_position_ends.dat'])

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from sTLReader1
sTLReader1Display = Show(sTLReader1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sTLReader1Display.Representation = 'Surface'
sTLReader1Display.AmbientColor = [0.0, 0.0, 1.0]
sTLReader1Display.ColorArrayName = ['POINTS', '']
sTLReader1Display.DiffuseColor = [0.0, 0.0, 1.0]
sTLReader1Display.OSPRayScaleFunction = 'PiecewiseFunction'
sTLReader1Display.SelectOrientationVectors = 'None'
sTLReader1Display.ScaleFactor = 1274.2
sTLReader1Display.SelectScaleArray = 'None'
sTLReader1Display.GlyphType = 'Arrow'
sTLReader1Display.GlyphTableIndexArray = 'None'
sTLReader1Display.GaussianRadius = 63.71
sTLReader1Display.SetScaleArray = ['POINTS', '']
sTLReader1Display.ScaleTransferFunction = 'PiecewiseFunction'
sTLReader1Display.OpacityArray = ['POINTS', '']
sTLReader1Display.OpacityTransferFunction = 'PiecewiseFunction'
sTLReader1Display.DataAxesGrid = 'GridAxesRepresentation'
sTLReader1Display.PolarAxes = 'PolarAxesRepresentation'

# show data from sTLReader2
sTLReader2Display = Show(sTLReader2, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sTLReader2Display.Representation = 'Surface'
sTLReader2Display.AmbientColor = [0.0, 0.6666666666666666, 0.0]
sTLReader2Display.ColorArrayName = ['POINTS', '']
sTLReader2Display.DiffuseColor = [0.0, 0.6666666666666666, 0.0]
sTLReader2Display.OSPRayScaleFunction = 'PiecewiseFunction'
sTLReader2Display.SelectOrientationVectors = 'None'
sTLReader2Display.ScaleFactor = 1306.177001953125
sTLReader2Display.SelectScaleArray = 'None'
sTLReader2Display.GlyphType = 'Arrow'
sTLReader2Display.GlyphTableIndexArray = 'None'
sTLReader2Display.GaussianRadius = 65.30885009765625
sTLReader2Display.SetScaleArray = ['POINTS', '']
sTLReader2Display.ScaleTransferFunction = 'PiecewiseFunction'
sTLReader2Display.OpacityArray = ['POINTS', '']
sTLReader2Display.OpacityTransferFunction = 'PiecewiseFunction'
sTLReader2Display.DataAxesGrid = 'GridAxesRepresentation'
sTLReader2Display.PolarAxes = 'PolarAxesRepresentation'

# show data from tecplotReader2
tecplotReader2Display = Show(tecplotReader2, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
tecplotReader2Display.Representation = 'Surface'
tecplotReader2Display.AmbientColor = [0.3333333333333333, 1.0, 1.0]
tecplotReader2Display.ColorArrayName = ['POINTS', '']
tecplotReader2Display.DiffuseColor = [0.3333333333333333, 1.0, 1.0]
tecplotReader2Display.LineWidth = 7.0
tecplotReader2Display.OSPRayScaleFunction = 'PiecewiseFunction'
tecplotReader2Display.SelectOrientationVectors = 'None'
tecplotReader2Display.ScaleFactor = 984.4389892578125
tecplotReader2Display.SelectScaleArray = 'None'
tecplotReader2Display.GlyphType = 'Arrow'
tecplotReader2Display.GlyphTableIndexArray = 'None'
tecplotReader2Display.GaussianRadius = 49.22194946289063
tecplotReader2Display.SetScaleArray = ['POINTS', '']
tecplotReader2Display.ScaleTransferFunction = 'PiecewiseFunction'
tecplotReader2Display.OpacityArray = ['POINTS', '']
tecplotReader2Display.OpacityTransferFunction = 'PiecewiseFunction'
tecplotReader2Display.DataAxesGrid = 'GridAxesRepresentation'
tecplotReader2Display.PolarAxes = 'PolarAxesRepresentation'
tecplotReader2Display.ScalarOpacityUnitDistance = 1854.2564182668152

# show data from tecplotReader4
tecplotReader4Display = Show(tecplotReader4, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
tecplotReader4Display.Representation = 'Points'
tecplotReader4Display.AmbientColor = [0.3333333333333333, 1.0, 1.0]
tecplotReader4Display.ColorArrayName = ['POINTS', '']
tecplotReader4Display.DiffuseColor = [0.3333333333333333, 1.0, 1.0]
tecplotReader4Display.PointSize = 20.0
tecplotReader4Display.RenderPointsAsSpheres = 1
tecplotReader4Display.OSPRayScaleFunction = 'PiecewiseFunction'
tecplotReader4Display.SelectOrientationVectors = 'None'
tecplotReader4Display.ScaleFactor = 970.1473876953125
tecplotReader4Display.SelectScaleArray = 'None'
tecplotReader4Display.GlyphType = 'Arrow'
tecplotReader4Display.GlyphTableIndexArray = 'None'
tecplotReader4Display.GaussianRadius = 48.507369384765624
tecplotReader4Display.SetScaleArray = ['POINTS', '']
tecplotReader4Display.ScaleTransferFunction = 'PiecewiseFunction'
tecplotReader4Display.OpacityArray = ['POINTS', '']
tecplotReader4Display.OpacityTransferFunction = 'PiecewiseFunction'
tecplotReader4Display.DataAxesGrid = 'GridAxesRepresentation'
tecplotReader4Display.PolarAxes = 'PolarAxesRepresentation'
tecplotReader4Display.ScalarOpacityUnitDistance = 14264.964869927227

# show data from sTLReader5
sTLReader5Display = Show(sTLReader5, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sTLReader5Display.Representation = 'Surface'
sTLReader5Display.AmbientColor = [0.3333333333333333, 1.0, 1.0]
sTLReader5Display.ColorArrayName = ['POINTS', '']
sTLReader5Display.DiffuseColor = [0.3333333333333333, 1.0, 1.0]
sTLReader5Display.OSPRayScaleFunction = 'PiecewiseFunction'
sTLReader5Display.SelectOrientationVectors = 'None'
sTLReader5Display.ScaleFactor = 1014.7530517578125
sTLReader5Display.SelectScaleArray = 'None'
sTLReader5Display.GlyphType = 'Arrow'
sTLReader5Display.GlyphTableIndexArray = 'None'
sTLReader5Display.GaussianRadius = 50.73765258789063
sTLReader5Display.SetScaleArray = ['POINTS', '']
sTLReader5Display.ScaleTransferFunction = 'PiecewiseFunction'
sTLReader5Display.OpacityArray = ['POINTS', '']
sTLReader5Display.OpacityTransferFunction = 'PiecewiseFunction'
sTLReader5Display.DataAxesGrid = 'GridAxesRepresentation'
sTLReader5Display.PolarAxes = 'PolarAxesRepresentation'

# show data from tecplotReader3
tecplotReader3Display = Show(tecplotReader3, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
tecplotReader3Display.Representation = 'Surface'
tecplotReader3Display.AmbientColor = [1.0, 0.6666666666666666, 1.0]
tecplotReader3Display.ColorArrayName = ['POINTS', '']
tecplotReader3Display.DiffuseColor = [1.0, 0.6666666666666666, 1.0]
tecplotReader3Display.LineWidth = 7.0
tecplotReader3Display.OSPRayScaleFunction = 'PiecewiseFunction'
tecplotReader3Display.SelectOrientationVectors = 'None'
tecplotReader3Display.ScaleFactor = 915.258544921875
tecplotReader3Display.SelectScaleArray = 'None'
tecplotReader3Display.GlyphType = 'Arrow'
tecplotReader3Display.GlyphTableIndexArray = 'None'
tecplotReader3Display.GaussianRadius = 45.76292724609375
tecplotReader3Display.SetScaleArray = ['POINTS', '']
tecplotReader3Display.ScaleTransferFunction = 'PiecewiseFunction'
tecplotReader3Display.OpacityArray = ['POINTS', '']
tecplotReader3Display.OpacityTransferFunction = 'PiecewiseFunction'
tecplotReader3Display.DataAxesGrid = 'GridAxesRepresentation'
tecplotReader3Display.PolarAxes = 'PolarAxesRepresentation'
tecplotReader3Display.ScalarOpacityUnitDistance = 1702.6349975323906

# show data from tecplotReader5
tecplotReader5Display = Show(tecplotReader5, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
tecplotReader5Display.Representation = 'Points'
tecplotReader5Display.AmbientColor = [1.0, 0.6666666666666666, 1.0]
tecplotReader5Display.ColorArrayName = ['POINTS', '']
tecplotReader5Display.DiffuseColor = [1.0, 0.6666666666666666, 1.0]
tecplotReader5Display.PointSize = 20.0
tecplotReader5Display.RenderPointsAsSpheres = 1
tecplotReader5Display.OSPRayScaleFunction = 'PiecewiseFunction'
tecplotReader5Display.SelectOrientationVectors = 'None'
tecplotReader5Display.ScaleFactor = 851.720703125
tecplotReader5Display.SelectScaleArray = 'None'
tecplotReader5Display.GlyphType = 'Arrow'
tecplotReader5Display.GlyphTableIndexArray = 'None'
tecplotReader5Display.GaussianRadius = 42.58603515625
tecplotReader5Display.SetScaleArray = ['POINTS', '']
tecplotReader5Display.ScaleTransferFunction = 'PiecewiseFunction'
tecplotReader5Display.OpacityArray = ['POINTS', '']
tecplotReader5Display.OpacityTransferFunction = 'PiecewiseFunction'
tecplotReader5Display.DataAxesGrid = 'GridAxesRepresentation'
tecplotReader5Display.PolarAxes = 'PolarAxesRepresentation'
tecplotReader5Display.ScalarOpacityUnitDistance = 12657.96145291931

# show data from sTLReader3
sTLReader3Display = Show(sTLReader3, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sTLReader3Display.Representation = 'Surface'
sTLReader3Display.AmbientColor = [1.0, 0.6666666666666666, 1.0]
sTLReader3Display.ColorArrayName = ['POINTS', '']
sTLReader3Display.DiffuseColor = [1.0, 0.6666666666666666, 1.0]
sTLReader3Display.OSPRayScaleFunction = 'PiecewiseFunction'
sTLReader3Display.SelectOrientationVectors = 'None'
sTLReader3Display.ScaleFactor = 915.258544921875
sTLReader3Display.SelectScaleArray = 'None'
sTLReader3Display.GlyphType = 'Arrow'
sTLReader3Display.GlyphTableIndexArray = 'None'
sTLReader3Display.GaussianRadius = 45.76292724609375
sTLReader3Display.SetScaleArray = ['POINTS', '']
sTLReader3Display.ScaleTransferFunction = 'PiecewiseFunction'
sTLReader3Display.OpacityArray = ['POINTS', '']
sTLReader3Display.OpacityTransferFunction = 'PiecewiseFunction'
sTLReader3Display.DataAxesGrid = 'GridAxesRepresentation'
sTLReader3Display.PolarAxes = 'PolarAxesRepresentation'

# show data from tecplotReader1
tecplotReader1Display = Show(tecplotReader1, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
tecplotReader1Display.Representation = 'Surface'
tecplotReader1Display.AmbientColor = [0.0, 1.0, 0.0]
tecplotReader1Display.ColorArrayName = ['POINTS', '']
tecplotReader1Display.DiffuseColor = [0.0, 1.0, 0.0]
tecplotReader1Display.LineWidth = 7.0
tecplotReader1Display.OSPRayScaleFunction = 'PiecewiseFunction'
tecplotReader1Display.SelectOrientationVectors = 'None'
tecplotReader1Display.ScaleFactor = 1020.4434814453125
tecplotReader1Display.SelectScaleArray = 'None'
tecplotReader1Display.GlyphType = 'Arrow'
tecplotReader1Display.GlyphTableIndexArray = 'None'
tecplotReader1Display.GaussianRadius = 51.02217407226563
tecplotReader1Display.SetScaleArray = ['POINTS', '']
tecplotReader1Display.ScaleTransferFunction = 'PiecewiseFunction'
tecplotReader1Display.OpacityArray = ['POINTS', '']
tecplotReader1Display.OpacityTransferFunction = 'PiecewiseFunction'
tecplotReader1Display.DataAxesGrid = 'GridAxesRepresentation'
tecplotReader1Display.PolarAxes = 'PolarAxesRepresentation'
tecplotReader1Display.ScalarOpacityUnitDistance = 1728.5119508844452

# show data from tecplotReader6
tecplotReader6Display = Show(tecplotReader6, renderView1, 'StructuredGridRepresentation')

# trace defaults for the display properties.
tecplotReader6Display.Representation = 'Points'
tecplotReader6Display.AmbientColor = [0.0, 1.0, 0.0]
tecplotReader6Display.ColorArrayName = ['POINTS', '']
tecplotReader6Display.DiffuseColor = [0.0, 1.0, 0.0]
tecplotReader6Display.PointSize = 20.0
tecplotReader6Display.RenderPointsAsSpheres = 1
tecplotReader6Display.OSPRayScaleFunction = 'PiecewiseFunction'
tecplotReader6Display.SelectOrientationVectors = 'None'
tecplotReader6Display.ScaleFactor = 953.6405517578125
tecplotReader6Display.SelectScaleArray = 'None'
tecplotReader6Display.GlyphType = 'Arrow'
tecplotReader6Display.GlyphTableIndexArray = 'None'
tecplotReader6Display.GaussianRadius = 47.68202758789062
tecplotReader6Display.SetScaleArray = ['POINTS', '']
tecplotReader6Display.ScaleTransferFunction = 'PiecewiseFunction'
tecplotReader6Display.OpacityArray = ['POINTS', '']
tecplotReader6Display.OpacityTransferFunction = 'PiecewiseFunction'
tecplotReader6Display.DataAxesGrid = 'GridAxesRepresentation'
tecplotReader6Display.PolarAxes = 'PolarAxesRepresentation'
tecplotReader6Display.ScalarOpacityUnitDistance = 13059.069220736674

# show data from sTLReader4
sTLReader4Display = Show(sTLReader4, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
sTLReader4Display.Representation = 'Surface'
sTLReader4Display.AmbientColor = [0.0, 1.0, 0.0]
sTLReader4Display.ColorArrayName = ['POINTS', '']
sTLReader4Display.DiffuseColor = [0.0, 1.0, 0.0]
sTLReader4Display.OSPRayScaleFunction = 'PiecewiseFunction'
sTLReader4Display.SelectOrientationVectors = 'None'
sTLReader4Display.ScaleFactor = 1020.4434814453125
sTLReader4Display.SelectScaleArray = 'None'
sTLReader4Display.GlyphType = 'Arrow'
sTLReader4Display.GlyphTableIndexArray = 'None'
sTLReader4Display.GaussianRadius = 51.02217407226563
sTLReader4Display.SetScaleArray = ['POINTS', '']
sTLReader4Display.ScaleTransferFunction = 'PiecewiseFunction'
sTLReader4Display.OpacityArray = ['POINTS', '']
sTLReader4Display.OpacityTransferFunction = 'PiecewiseFunction'
sTLReader4Display.DataAxesGrid = 'GridAxesRepresentation'
sTLReader4Display.PolarAxes = 'PolarAxesRepresentation'

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(tecplotReader1)
# ----------------------------------------------------------------