"""
High-performance collision detection for SANTA2025
Based on SA.py and sa_runner_original.cpp
"""

from shapely.strtree import STRtree


def check_collision_fast(new_poly, new_bounds, current_polys, current_bounds, exclude_idx=None):
    """
    Fast collision detection using AABB pre-filtering + shapely geometry checks

    Args:
        new_poly: New polygon to check
        new_bounds: Bounding box of new_poly (minx, miny, maxx, maxy)
        current_polys: List of existing polygons
        current_bounds: List of existing bounding boxes
        exclude_idx: Index to exclude (self-collision, optional)

    Returns:
        True if collision detected, False otherwise

    Note:
        - Touching (boundary contact) is allowed
        - Only area overlap is considered collision
    """
    minx, miny, maxx, maxy = new_bounds
    n_trees = len(current_polys)

    for k in range(n_trees):
        # Skip self
        if exclude_idx is not None and k == exclude_idx:
            continue

        # AABB pre-filter (fast rejection)
        ox1, oy1, ox2, oy2 = current_bounds[k]
        if maxx < ox1 or minx > ox2 or maxy < oy1 or miny > oy2:
            continue  # Bounding boxes don't overlap

        # Geometry check (slower but accurate)
        other = current_polys[k]

        # Allow touches (boundary contact), reject only area overlap
        # not disjoint = intersects or touches
        # not touches = either disjoint or overlaps
        # so: (not disjoint) and (not touches) = overlaps
        if (not new_poly.disjoint(other)) and (not new_poly.touches(other)):
            return True  # Collision detected

    return False  # No collision


def check_collision_strtree(new_poly, strtree, all_polys, exclude_idx=None):
    """
    Collision detection using STRtree spatial index

    Args:
        new_poly: New polygon to check
        strtree: STRtree spatial index of all_polys
        all_polys: List of all polygons
        exclude_idx: Index to exclude (optional)

    Returns:
        True if collision detected, False otherwise
    """
    # Query spatial index for candidates
    candidates = strtree.query(new_poly)

    for cand in candidates:
        # Handle different shapely versions
        if hasattr(cand, "geom_type"):
            # Shapely 1.x returns geometry objects
            other = cand
            if other is new_poly:
                continue

            # Find index if needed
            if exclude_idx is not None:
                try:
                    idx = all_polys.index(other)
                    if idx == exclude_idx:
                        continue
                except ValueError:
                    pass
        else:
            # Shapely 2.x returns indices
            idx = int(cand)
            if exclude_idx is not None and idx == exclude_idx:
                continue
            other = all_polys[idx]

        # Check overlap (allow touches)
        if (not new_poly.disjoint(other)) and (not new_poly.touches(other)):
            return True

    return False


def validate_no_overlaps(polygons, use_strtree=True):
    """
    Validate that no polygons overlap (touching is allowed)

    Args:
        polygons: List of shapely Polygon objects
        use_strtree: Use spatial index (faster for large groups)

    Returns:
        True if no overlaps, False if any overlap detected
    """
    if not polygons:
        return True

    if use_strtree and len(polygons) > 10:
        # Use spatial index for large groups
        strtree = STRtree(polygons)

        for i, poly in enumerate(polygons):
            candidates = strtree.query(poly)

            for cand in candidates:
                # Handle different shapely versions
                if hasattr(cand, "geom_type"):
                    other = cand
                    if other is poly:
                        continue
                else:
                    j = int(cand)
                    if j == i:
                        continue
                    other = polygons[j]

                # Check overlap
                if (not poly.disjoint(other)) and (not poly.touches(other)):
                    return False

        return True
    else:
        # Brute force for small groups
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                poly1 = polygons[i]
                poly2 = polygons[j]

                # Check overlap
                if (not poly1.disjoint(poly2)) and (not poly1.touches(poly2)):
                    return False

        return True


def get_collision_pairs(polygons):
    """
    Find all pairs of overlapping polygons (for debugging)

    Args:
        polygons: List of shapely Polygon objects

    Returns:
        List of (i, j) index pairs that have collision
    """
    if not polygons:
        return []

    collision_pairs = []
    strtree = STRtree(polygons)

    for i, poly in enumerate(polygons):
        candidates = strtree.query(poly)

        for cand in candidates:
            if hasattr(cand, "geom_type"):
                other = cand
                try:
                    j = polygons.index(other)
                except ValueError:
                    continue
            else:
                j = int(cand)

            # Only check each pair once (i < j)
            if j <= i:
                continue

            other = polygons[j]

            # Check overlap
            if (not poly.disjoint(other)) and (not poly.touches(other)):
                collision_pairs.append((i, j))

    return collision_pairs


# Optimized version using precomputed bounds
class CollisionChecker:
    """
    Reusable collision checker with cached bounds
    Use this for SA optimization where polygons change frequently
    """

    def __init__(self, initial_polys):
        """
        Initialize with initial polygon list

        Args:
            initial_polys: List of shapely Polygon objects
        """
        self.polys = list(initial_polys)
        self.bounds = [p.bounds for p in self.polys]
        self.n = len(self.polys)

    def update_polygon(self, idx, new_poly):
        """
        Update polygon at index

        Args:
            idx: Index to update
            new_poly: New polygon
        """
        self.polys[idx] = new_poly
        self.bounds[idx] = new_poly.bounds

    def check_collision(self, idx, test_poly, test_bounds):
        """
        Check if test_poly collides with any other polygon

        Args:
            idx: Index being tested (to exclude from check)
            test_poly: Test polygon
            test_bounds: Bounding box of test_poly

        Returns:
            True if collision, False otherwise
        """
        minx, miny, maxx, maxy = test_bounds

        for k in range(self.n):
            if k == idx:
                continue

            # AABB filter
            ox1, oy1, ox2, oy2 = self.bounds[k]
            if maxx < ox1 or minx > ox2 or maxy < oy1 or miny > oy2:
                continue

            # Geometry check
            other = self.polys[k]
            if (not test_poly.disjoint(other)) and (not test_poly.touches(other)):
                return True

        return False

    def validate_all(self):
        """
        Validate current state has no collisions

        Returns:
            True if no collisions, False otherwise
        """
        return validate_no_overlaps(self.polys, use_strtree=True)


if __name__ == '__main__':
    """Test collision detection"""
    from clusters import ChristmasTree, Dimer
    from decimal import Decimal

    print("Testing collision detection...")

    # Test 1: No collision
    print("\n=== Test 1: No Collision ===")
    tree1 = ChristmasTree('0', '0', '0')
    tree2 = ChristmasTree('2', '0', '0')

    polys = [tree1.polygon, tree2.polygon]
    result = validate_no_overlaps(polys)
    print(f"Trees at (0,0) and (2,0): {result} (expected True)")

    # Test 2: Collision
    print("\n=== Test 2: Collision ===")
    tree3 = ChristmasTree('0', '0', '0')
    tree4 = ChristmasTree('0.1', '0', '0')  # Too close!

    polys2 = [tree3.polygon, tree4.polygon]
    result2 = validate_no_overlaps(polys2)
    print(f"Trees at (0,0) and (0.1,0): {result2} (expected False)")

    # Test 3: Dimer collision
    print("\n=== Test 3: Dimer No Collision ===")
    dimer1 = Dimer(ChristmasTree('0', '0', '0'), ChristmasTree('0.35', '0.8', '180'))
    dimer2 = Dimer(ChristmasTree('2', '0', '0'), ChristmasTree('2.35', '0.8', '180'))

    all_polys = dimer1.get_polygons() + dimer2.get_polygons()
    result3 = validate_no_overlaps(all_polys)
    print(f"Two dimers at distance: {result3} (expected True)")

    # Test 4: Fast check
    print("\n=== Test 4: Fast Check ===")
    tree5 = ChristmasTree('0', '0', '0')
    tree6 = ChristmasTree('1', '0', '0')
    tree7 = ChristmasTree('2', '0', '0')

    current_polys = [tree5.polygon, tree6.polygon]
    current_bounds = [p.bounds for p in current_polys]

    # Test new position
    tree_test = ChristmasTree('0.1', '0', '0')
    collision = check_collision_fast(
        tree_test.polygon,
        tree_test.polygon.bounds,
        current_polys,
        current_bounds
    )
    print(f"New tree at (0.1, 0) vs existing: {collision} (expected True - collision)")

    tree_test2 = ChristmasTree('3', '0', '0')
    collision2 = check_collision_fast(
        tree_test2.polygon,
        tree_test2.polygon.bounds,
        current_polys,
        current_bounds
    )
    print(f"New tree at (3, 0) vs existing: {collision2} (expected False - no collision)")

    # Test 5: CollisionChecker class
    print("\n=== Test 5: CollisionChecker ===")
    checker = CollisionChecker([tree5.polygon, tree6.polygon])

    test_poly = tree_test.polygon
    test_bounds = test_poly.bounds
    collision3 = checker.check_collision(0, test_poly, test_bounds)
    print(f"Using CollisionChecker: {collision3} (expected True)")

    # Test 6: Performance test
    print("\n=== Test 6: Performance Test ===")
    import time

    # Create many trees
    n_test = 100
    test_trees = []
    for i in range(10):
        for j in range(10):
            tree = ChristmasTree(str(i * 1.2), str(j * 1.2), '0')
            test_trees.append(tree)

    test_polys = [t.polygon for t in test_trees]

    # Method 1: With STRtree
    start = time.time()
    for _ in range(100):
        result = validate_no_overlaps(test_polys, use_strtree=True)
    time_strtree = time.time() - start

    # Method 2: Brute force
    start = time.time()
    for _ in range(100):
        result = validate_no_overlaps(test_polys, use_strtree=False)
    time_brute = time.time() - start

    print(f"STRtree method: {time_strtree:.3f}s (100 iterations)")
    print(f"Brute force:    {time_brute:.3f}s (100 iterations)")
    print(f"Speedup:        {time_brute/time_strtree:.2f}x")

    print("\n All tests completed!")
