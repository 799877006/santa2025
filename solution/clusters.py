from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry import Point


getcontext().prec = 50
scale_factor = Decimal('1e18')

class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)  # 树中心 x
        self.center_y = Decimal(center_y)  # 树中心 y
        self.angle = Decimal(angle)  # 旋转角度（度）
        self.polygon = self._create_polygon()  # 构造多边形

    def _create_polygon(self):
        trunk_w = Decimal('0.15'); trunk_h = Decimal('0.2')  # 树干宽、高
        base_w = Decimal('0.7'); base_y = Decimal('0.0')     # 底层宽、y
        mid_w = Decimal('0.4'); tier_2_y = Decimal('0.25')   # 中层宽、y
        top_w = Decimal('0.25'); tier_1_y = Decimal('0.5')   # 上层宽、y
        tip_y = Decimal('0.8'); trunk_bottom_y = -trunk_h    # 树尖 y、树干底 y

        # 以原点为中心的树形轮廓，所有坐标都放大 scale_factor
        initial_polygon = Polygon([
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))  # 围绕原点旋转
        return affinity.translate(  # 平移到指定中心
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor)
        )

    # def clone(self) -> "ChristmasTree":
    #     new_tree = ChristmasTree.__new__(ChristmasTree)  # 不调用 __init__，直接复制字段
    #     new_tree.center_x = self.center_x
    #     new_tree.center_y = self.center_y
    #     new_tree.angle = self.angle
    #     new_tree.polygon = self.polygon
    #     return new_tree
    
    def get_polygons(self):
        return [self.polygon]


class Dimer():
    '''
    Dimer's center is the average of the trees in it
    '''
    
    def __init__(self,tree_a:ChristmasTree,tree_b:ChristmasTree):
        self.tree_a = tree_a
        self.tree_b = tree_b
    
    @property
    def center_x(self):
        return (self.tree_a.center_x + self.tree_b.center_x) / Decimal('2')
    
    @property
    def center_y(self):
        return (self.tree_a.center_y + self.tree_b.center_y) / Decimal('2')
    
    def get_polygons(self):
        return [self.tree_a.polygon,self.tree_b.polygon]

    def rotate(self,angle:Decimal):
        cx = self.center_x
        cy = self.center_y

        cx_scaled = float(cx *scale_factor)
        cy_scaled = float(cy *scale_factor)
        angle_float = float(angle)
        for tree in [self.tree_a,self.tree_b]:
            tree.polygon = affinity.rotate(tree.polygon,angle_float,origin=(cx_scaled,cy_scaled))
            
            #get the center(pivot) of the tree after rotation
            current_center = Point(float(tree.center_x* scale_factor),float(tree.center_y * scale_factor))
            rotated_center = affinity.rotate(current_center,angle_float,origin=(cx_scaled,cy_scaled))
            tree.center_x = Decimal(str(rotated_center.x / float(scale_factor)))
            tree.center_y = Decimal(str(rotated_center.y / float(scale_factor)))
            tree.angle = tree.angle + angle
        
    

    def translate(self,dx:Decimal,dy:Decimal):
        dx_scaled = float(dx * scale_factor)
        dy_scaled = float(dy * scale_factor)
        for tree in [self.tree_a,self.tree_b]:
            tree.polygon = affinity.translate(tree.polygon,xoff=dx_scaled,yoff=dy_scaled)
            tree.center_x += dx
            tree.center_y += dy

