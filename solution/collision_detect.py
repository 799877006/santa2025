"""
High-performance collision detection for SANTA2025
Based on SA.py and sa_runner_original.cpp
"""

from shapely.strtree import STRtree


def check_collision(new_poly, strtree=None, current_polys=None, current_bounds=None, exclude_idx=None):
    """
    Unified collision detection function.
    
    Strategy:
    1. Fast Phase (Broad Phase):
       - If strtree provided: Use spatial index query (O(log N))
       - If list provided: Use AABB pre-filtering (O(N))
       - Result: Set of candidate polygons that *might* collide
    
    2. Strict Phase (Narrow Phase):
       - For each candidate, perform precise geometry check (disjoint/touches)
       - O(k) where k is small number of candidates
       
    Args:
        new_poly: New polygon to check
        strtree: STRtree spatial index (optional, faster for large N)
        current_polys: List of existing polygons (required if strtree not provided)
        current_bounds: List of existing bounding boxes (optional, for list mode optimization)
        exclude_idx: Index to exclude (self-collision check)

    Returns:
        True if collision detected (overlap), False otherwise
    """
    
    # === Fast Phase: Candidate Selection ===
    candidates = []
    use_indices = False
    
    if strtree is not None:
        # STRtree mode (Fastest for large N)
        query_result = strtree.query(new_poly)
        
        # Normalize STRtree results (Shapely 1.x vs 2.x)
        for item in query_result:
            if hasattr(item, "geom_type"):
                # Shapely 1.x: returns geometry objects
                if item is new_poly: continue
                candidates.append(item)
            else:
                # Shapely 2.x: returns indices
                idx = int(item)
                if exclude_idx is not None and idx == exclude_idx: continue
                if current_polys:
                    candidates.append(current_polys[idx])
                else:
                    # If we only have strtree but no poly list access by index...
                    # This case shouldn't happen if caller is correct, but let's handle
                    pass 

    elif current_polys is not None:
        # List Iteration mode (O(N), good for small N or dynamic updates)
        minx, miny, maxx, maxy = new_poly.bounds
        
        if current_bounds:
            # Optimized list mode with precomputed bounds
            n = len(current_polys)
            for k in range(n):
                if exclude_idx is not None and k == exclude_idx: continue
                
                # AABB Check
                ox1, oy1, ox2, oy2 = current_bounds[k]
                if maxx < ox1 or minx > ox2 or maxy < oy1 or miny > oy2:
                    continue
                
                candidates.append(current_polys[k])
        else:
            # Slow list mode (compute bounds on fly)
            for k, poly in enumerate(current_polys):
                if exclude_idx is not None and k == exclude_idx: continue
                
                # Check AABB on fly
                p_minx, p_miny, p_maxx, p_maxy = poly.bounds
                if maxx < p_minx or minx > p_maxx or maxy < p_miny or miny > p_maxy:
                    continue
                    
                candidates.append(poly)
    else:
        raise ValueError("Must provide either strtree or current_polys")

    # === Strict Phase: Geometry Check ===
    for other in candidates:
        # Check specific exclusion for Shapely 1.x object matching
        if exclude_idx is not None and current_polys:
             try:
                 # Note: This is slow, but only needed if exclude_idx is used with object-based candidates
                 if current_polys.index(other) == exclude_idx:
                     continue
             except ValueError:
                 pass

        # Precise Intersection Test
        # allow touches (boundary contact), reject only area overlap
        if (not new_poly.disjoint(other)) and (not new_poly.touches(other)):
            return True

    return False

# Legacy aliases for backward compatibility
def check_collision_fast(new_poly, new_bounds, current_polys, current_bounds, exclude_idx=None):
    return check_collision(new_poly, current_polys=current_polys, current_bounds=current_bounds, exclude_idx=exclude_idx)

def check_collision_strtree(new_poly, strtree, all_polys, exclude_idx=None):
    return check_collision(new_poly, strtree=strtree, current_polys=all_polys, exclude_idx=exclude_idx)


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
            # Pass exclude_idx to prevent self-collision
            if check_collision(poly, strtree=strtree, current_polys=polygons, exclude_idx=i):
                return False
        return True
    else:
        # Brute force for small groups
        current_bounds = [p.bounds for p in polygons]
        for i, poly in enumerate(polygons):
             # Only check against others (could optimize to i+1..n but reusing general function)
             if check_collision(poly, current_polys=polygons, current_bounds=current_bounds, exclude_idx=i):
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
        # Delegate to unified function
        return check_collision(test_poly, current_polys=self.polys, current_bounds=self.bounds, exclude_idx=idx)

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

    # Test 4: Fast check (via Unified check_collision)
    print("\n=== Test 4: Unified Check (List Mode) ===")
    tree5 = ChristmasTree('0', '0', '0')
    tree6 = ChristmasTree('1', '0', '0')
    tree7 = ChristmasTree('2', '0', '0')

    current_polys = [tree5.polygon, tree6.polygon]
    current_bounds = [p.bounds for p in current_polys]

    # Test new position
    tree_test = ChristmasTree('0.1', '0', '0')
    collision = check_collision(
        tree_test.polygon,
        current_polys=current_polys,
        current_bounds=current_bounds
    )
    print(f"New tree at (0.1, 0) vs existing: {collision} (expected True - collision)")

    tree_test2 = ChristmasTree('3', '0', '0')
    collision2 = check_collision(
        tree_test2.polygon,
        current_polys=current_polys,
        current_bounds=current_bounds
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

    # Method 1: With STRtree (Unified)
    start = time.time()
    for _ in range(100):
        result = validate_no_overlaps(test_polys, use_strtree=True)
    time_strtree = time.time() - start

    # Method 2: Brute force (Unified list mode)
    start = time.time()
    for _ in range(100):
        result = validate_no_overlaps(test_polys, use_strtree=False)
    time_brute = time.time() - start

    print(f"STRtree method: {time_strtree:.3f}s (100 iterations)")
    print(f"Brute force:    {time_brute:.3f}s (100 iterations)")
    print(f"Speedup:        {time_brute/time_strtree:.2f}x")

    print("\nAll tests completed!")
