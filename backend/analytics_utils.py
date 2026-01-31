def on_segment(p, r):
    """Check if point r lies on segment 'p-r' (collinear assumed beforehand)."""
    if (r[0] <= max(p[0], r[0]) and r[0] >= min(p[0], r[0]) and
        r[1] <= max(p[1], r[1]) and r[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    """
    0 -> Collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def do_intersect(p1, q1, p2, q2):
    """
    Check if segment p1-q1 intersects with segment p2-q2.
    Points are tuples or lists [x, y].
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1, p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2): return True
    # p1, q1, q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2): return True
    # p2, q2, p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1): return True
    # p2, q2, q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1): return True

    return False
