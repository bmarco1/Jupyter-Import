points = [(-2, -4), (0, -2), (-1, 0), (3, -5), (-2, -3), (3, 2)]

def closest(points, k):
    using points, create points with d
    using points_with_d, create a max heap with the first k items
    call this max heap MH
    for p in points_with_d[k...n-1]:
        if p.distance < MH.getMax():
            replace MH's current max points with p
print all points in MH
