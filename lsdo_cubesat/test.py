# 1D

geo = TriangulationGeometry(filename='geom.tri')

rib_pt1 = geo.project_point([1, 2, 3])
rib_pt2 = geo.project_point([1, 2, 3])

rib_upper = GeoCurve(rib_pt1, rib_pt2)
rib_lower = GeoCurve(rib_pt3, rib_pt4)

rib = InternalTB(rib_upper, rib_lower)
