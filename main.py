import random
import math
import matplotlib.pyplot as plt
import functools
import time


"""Credit to Tom Switzer for his original implementation of Chan's/Graham Scan/Jarvis March Algorithms https://gist.github.com/tixxit/252229"""


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# sort point set by x-coordinate
def sort_points(points):
    return sorted(points, key=lambda point: point.x)


# returns -1 if CW, 0 if Collinear, and 1 if CCW
def orientation(a, b, c):
    result = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
    if result < 0:
        return -1
    elif result > 0:
        return 1
    return 0


# returns the squared Euclidean distance between p and q
def dist(a, b):
    dx, dy = b.x - a.x, b.y - a.y
    return dx*dx + dy*dy


# returns the next point on the convex hull CCW from a point
def next_hull_pt(points, point):
    q = point
    for r in points:
        t = orientation(point, q, r)
        if t == -1 or t == 0 and dist(point, r) > dist(point, q):
            q = r
    return q


# graham scan helper
def keep_left(hull, r):
    while len(hull) > 1 and orientation(hull[-2], hull[-1], r) != 1:
        hull.pop()
    if not len(hull) or hull[-1] != r:
        hull.append(r)
    return hull


# generates a dataset
def generate_dataset(amount, mode, min_x, min_y, max_x, max_y):
    result = []
    match mode:
        case 0:  # random
            for i in range(amount):
                dupe = True
                x = 0
                y = 0
                while dupe:
                    x = random.randint(min_x, max_x)
                    y = random.randint(min_y, max_y)
                    dupe = False
                    for point in result:
                        if x == point.x and y == point.y:
                            dupe = True
                            break
                result.append(Point(x, y))
            return result
        case 1:  # convex (along a circle)
            radius = min((max_x-min_x)/2, (max_y-min_y)/2)
            center = Point(min_x+radius, min_y+radius)
            for i in range(amount):
                dupe = True
                x = 0
                y = 0
                while dupe:
                    x = random.randint(min_x, max_x)
                    y_sign = [-1, 1][random.randrange(2)]
                    y = center.y + y_sign*(math.sqrt(pow(radius, 2)-pow(x-center.x, 2)))
                    dupe = False
                    for point in result:
                        if x == point.x and y == point.y:
                            dupe = True
                            break
                result.append(Point(x, y))
            return result
        case 2:  # bounding box
            width = max_x-min_x
            height = max_y-min_y
            for i in range(amount-4):
                dupe = True
                x = 0
                y = 0
                while dupe:
                    x = random.randint(round(min_x+width/50), round(max_x-width/50))
                    y = random.randint(round(min_y+height/50), round(max_y-height/50))
                    dupe = False
                    for point in result:
                        if x == point.x and y == point.y:
                            dupe = True
                            break
                result.append(Point(x, y))
            result.append(Point(min_x+1, min_y))
            result.append(Point(max_x, max_y))
            result.append(Point(max_x-1, min_y))
            result.append(Point(min_x, max_y))
            return result
        case 3:  # two clusters
            radius = round(min((max_x-min_x)/4, (max_y-min_y)/4))
            center1 = Point(min_x+radius, min_y+radius)
            center2 = Point(max_x-radius, max_y-radius)
            for i in range(amount):
                cluster = random.randint(0, 1)
                center = center1 if cluster == 0 else center2
                dupe = True
                x = 0
                y = 0
                while dupe:
                    x = random.randint(center.x-radius, center.x+radius)
                    y_sign = [-1, 1][random.randrange(2)]
                    y = center.y + y_sign * random.randint(0, round(math.sqrt(pow(radius, 2) - pow(x - center.x, 2))))
                    dupe = False
                    for point in result:
                        if x == point.x and y == point.y:
                            dupe = True
                            break
                result.append(Point(x, y))
            return result


# Return the index of the point in hull that the right tangent line from p to hull touches.
def right_tangent(hull, point):
    l, r = 0, len(hull)
    l_prev = orientation(point, hull[0], hull[-1])
    l_next = orientation(point, hull[0], hull[int((l + 1) % r)])
    while l < r:
        c = (l + r) / 2
        c_prev = orientation(point, hull[int(c)], hull[int(abs((c - 1) % len(hull)))])
        c_next = orientation(point, hull[int(c)], hull[int((c + 1) % len(hull))])
        c_side = orientation(point, hull[int(l)], hull[int(c)])
        if c_prev != -1 and c_next != -1:
            return c
        elif c_side == 1 and (l_next == -1 or l_prev == l_next) or c_side == -1 and c_prev == -1:
            r = c  # Tangent touches left chain
        else:
            l = c + 1  # Tangent touches right chain
            l_prev = -c_next  # Switch sides
            l_next = orientation(point, hull[int(l)], hull[int((l + 1) % len(hull))])
    return l


# Returns the hull, point index pair that is minimal.
def min_hull_pt_pair(hulls):
    h, p = 0, 0
    for i in range(len(hulls)):
        for j in range(len(hulls[i])):
            if hulls[i][j].x < hulls[h][p].x:
                h, p = int(i), int(j)
    return h, p


# Returns the (hull, point) index pair of the next point in the convex hull.
def next_hull_pt_pair(hulls, pair, opt):
    p = hulls[int(pair[0])][int(pair[1])]
    next = (pair[0], (pair[1] + 1) % len(hulls[pair[0]]))
    for h in (i for i in range(len(hulls)) if i != pair[0]):
        s = right_tangent(hulls[h], p)
        q, r = hulls[int(next[0])][int(next[1])], hulls[h][int(s)]
        t = orientation(p, q, r)
        if t == -1 or t == 0 and dist(p, r) > dist(p, q):
            next = (h, s)
    return next


# graham scan algorithm (returns convex hull)
def graham_scan(points):
    dupe = sort_points(points)
    lower = functools.reduce(keep_left, dupe, [])
    upper = functools.reduce(keep_left, reversed(dupe), [])
    result = lower.extend(upper[i] for i in range(1, len(upper) - 1)) or lower
    return result


# jarvis march algorithm (returns convex hull)
def jarvis_march(points):
    left = points[0]
    for point in points:
        if point.x < left.x:
            left = point
    result = [left]
    for point in result:
        q = next_hull_pt(points, point)
        if q != result[0]:
            result.append(q)
    return result


# chans algorithm with no optimizations (returns convex hull)
def chans_algorithm(points, i1=False, i2=False, i3=False, i4=False, i5=False):
    if i4:
        for H in (1 << (t*t) for t in range(2, len(points))):
            m = int(min(H*math.log(10, H), len(points))) if i2 else H
            hulls = []
            for i in range(0, len(points), m):
                hulls.append(graham_scan(points[i:i+m]))
            hull = [min_hull_pt_pair(hulls)]
            for throw_away in range(m):
                p = next_hull_pt_pair(hulls, hull[-1], i5)
                if p == hull[0]:
                    return [hulls[int(h)][int(i)] for h, i in hull]
                hull.append(p)
            if i1:
                points = []
                [points.extend(x) for x in hulls]
    else:
        for H in (1 << (1 << t) for t in range(len(points))):
            m = int(min(H*math.log(10, H), len(points))) if i2 else H
            hulls = []
            for i in range(0, len(points), m):
                hulls.append(graham_scan(points[i:i+m]))
            hull = [min_hull_pt_pair(hulls)]
            for throw_away in range(m):
                p = next_hull_pt_pair(hulls, hull[-1], i5)
                if p == hull[0]:
                    return [hulls[int(h)][int(i)] for h, i in hull]
                hull.append(p)
            if i1:
                points = []
                [points.extend(x) for x in hulls]


lst1 = []
lst2 = []
point_set = generate_dataset(10000, 0, 1, 1, 1000000, 1000000)
for p in point_set:
    lst1.append(p.x)
    lst2.append(p.y)
plt.figure(0)
plt.plot(lst1, lst2, linestyle='none', markersize=1.0, marker='.')

lst1.clear()
lst2.clear()
chan = chans_algorithm(point_set, True, True, True, True, True)
for p in chan:
    lst1.append(p.x)
    lst2.append(p.y)
lst1.append(chan[0].x)
lst2.append(chan[0].y)
plt.plot(lst1, lst2, markersize=1.0, marker='.', color='red')

lst1.clear()
lst2.clear()
point_set = generate_dataset(10000, 1, 1, 1, 1000000, 1000000)
for p in point_set:
    lst1.append(p.x)
    lst2.append(p.y)
plt.figure(1)
plt.plot(lst1, lst2, linestyle='none', markersize=1.0, marker='.')

lst1.clear()
lst2.clear()
chan = chans_algorithm(point_set, True, True, True, True, True)
for p in chan:
    lst1.append(p.x)
    lst2.append(p.y)
lst1.append(chan[0].x)
lst2.append(chan[0].y)
plt.plot(lst1, lst2, markersize=1.0, marker='.', color='red')

lst1.clear()
lst2.clear()
point_set = generate_dataset(10000, 2, 1, 1, 1000000, 1000000)
for p in point_set:
    lst1.append(p.x)
    lst2.append(p.y)
plt.figure(2)
plt.plot(lst1, lst2, linestyle='none', markersize=1.0, marker='.')

lst1.clear()
lst2.clear()
chan = chans_algorithm(point_set, True, True, True, True, True)
for p in chan:
    lst1.append(p.x)
    lst2.append(p.y)
lst1.append(chan[0].x)
lst2.append(chan[0].y)
plt.plot(lst1, lst2, markersize=1.0, marker='.', color='red')


lst1.clear()
lst2.clear()
point_set = generate_dataset(10000, 3, 1, 1, 1000000, 1000000)
for p in point_set:
    lst1.append(p.x)
    lst2.append(p.y)
plt.figure(3)
plt.plot(lst1, lst2, linestyle='none', markersize=1.0, marker='.')
lst1.clear()
lst2.clear()
chan = chans_algorithm(point_set, True, True, True, True, True)
for p in chan:
    lst1.append(p.x)
    lst2.append(p.y)
lst1.append(chan[0].x)
lst2.append(chan[0].y)
plt.plot(lst1, lst2, markersize=1.0, marker='.', color='red')

distributions = [100, 500, 1000, 5000, 10000, 50000, 100000]
point_sets = [[], [], [], []]
for i in range(4):
    for distribution in distributions:
        point_sets[i].append(generate_dataset(distribution, i, 0, 0, 1000000, 1000000))

for j in range(4):
    plt.figure(4+j)
    lst1.clear()
    lst2.clear()
    for i in range(len(distributions)):
        st = time.time()
        graham_scan(point_sets[j][i])
        lst2.append(time.time()-st)
        lst1.append(distributions[i])
    plt.plot(lst1, lst2, markersize=1.0, marker='.', color='red')
    lst1.clear()
    lst2.clear()
    for i in range(len(distributions)):
        st = time.time()
        jarvis_march(point_sets[j][i])
        lst2.append(time.time()-st)
        lst1.append(distributions[i])
    plt.plot(lst1, lst2, markersize=1.0, marker='.', color='orange')
    lst1.clear()
    lst2.clear()
    for i in range(len(distributions)):
        st = time.time()
        chans_algorithm(point_sets[j][i])
        lst2.append(time.time()-st)
        lst1.append(distributions[i])
    plt.plot(lst1, lst2, markersize=1.0, marker='.', color='yellow')
    lst1.clear()
    lst2.clear()
    for i in range(len(distributions)):
        st = time.time()
        chans_algorithm(point_sets[j][i], i1=True)
        lst2.append(time.time()-st)
        lst1.append(distributions[i])
    plt.plot(lst1, lst2, markersize=1.0, marker='.', color='green')
    lst1.clear()
    lst2.clear()
    for i in range(len(distributions)):
        st = time.time()
        chans_algorithm(point_sets[j][i], i2=True)
        lst2.append(time.time()-st)
        lst1.append(distributions[i])
    plt.plot(lst1, lst2, markersize=1.0, marker='.', color='blue')
    lst1.clear()
    lst2.clear()
    for i in range(len(distributions)):
        st = time.time()
        chans_algorithm(point_sets[j][i], i4=True)
        lst2.append(time.time()-st)
        lst1.append(distributions[i])
    plt.plot(lst1, lst2, markersize=1.0, marker='.', color='purple')

plt.show()
