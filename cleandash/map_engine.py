# map_engine.py
import networkx as nx
import json
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import *

class MapEngine:
    def __init__(self):
        self.G = nx.Graph()
        self.signs = []
        self.svg_w, self.svg_h = 600, 600
        self.pil_bg = Image.new('RGB', (600, 600), color='#1e1e1e')
        self.ppm_x = 1.0
        self.ppm_y = 1.0
        self.node_pixels = {}
        
        self._load_map_and_graph()
        self.load_signs()

    def _load_map_and_graph(self):
        if not os.path.exists(GRAPH_FILE):
            self.G.add_node("1", x=5.0, y=5.0)
            self.G.add_node("2", x=15.0, y=5.0)
            self.G.add_edge("1", "2")
        else:
            self.G = nx.read_graphml(GRAPH_FILE)

        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            if os.path.exists(SVG_FILE):
                d = svg2rlg(SVG_FILE)
                s = 600 / d.width
                d.width *= s; d.height *= s; d.scale(s, s)
                self.pil_bg = renderPM.drawToPIL(d, bg=0x111111)
                self.svg_w, self.svg_h = int(d.width), int(d.height)
        except BaseException as e:
            self.svg_w, self.svg_h = 600, 600
            self.pil_bg = Image.new('RGB', (self.svg_w, self.svg_h), color='#1e1e1e')
            draw = ImageDraw.Draw(self.pil_bg)
            draw.text((20, 20), "Fallback Map Background\n(Missing Track.svg)", fill="gray")

        self.ppm_x = self.svg_w / REAL_WIDTH_M
        self.ppm_y = self.svg_h / REAL_HEIGHT_M
        self.node_pixels = {n: self.to_pixel(float(d.get('x', 0)), float(d.get('y', 0))) for n, d in self.G.nodes(data=True)}

    def to_pixel(self, x, y): 
        return int((float(x)*self.ppm_x*FINAL_SCALE_X)+FINAL_OFF_X), int(self.svg_h-((float(y)*self.ppm_y*FINAL_SCALE_Y)+FINAL_OFF_Y))
    
    def to_meter(self, x, y): 
        return (x-FINAL_OFF_X)/(self.ppm_x*FINAL_SCALE_X), (self.svg_h-y-FINAL_OFF_Y)/(self.ppm_y*FINAL_SCALE_Y)

    def save_signs(self):
        with open(SIGNS_DB_FILE, 'w') as f: json.dump(self.signs, f)
        
    def load_signs(self):
        if os.path.exists(SIGNS_DB_FILE):
            with open(SIGNS_DB_FILE, 'r') as f: self.signs = json.load(f)
            migration_map = {
                "Stop": "stop-sign", "Crosswalk": "crosswalk-sign", "Priority": "priority-sign",
                "Parking": "parking-sign", "Highway Entry": "highway-entry-sign",
                "Highway Exit": "highway-exit-sign", "Pedestrian": "pedestrian",
                "Traffic Light": "traffic-light", "Roundabout": "roundabout-sign",
                "Oneway": "oneway-sign", "No Entry": "noentry-sign"
            }
            for s in self.signs:
                if s['type'] in migration_map: s['type'] = migration_map[s['type']]

    def calc_path_nodes(self, start_node, end_node, pass_node=None):
        try:
            if pass_node:
                p1 = nx.shortest_path(self.G, start_node, pass_node)
                p2 = nx.shortest_path(self.G, pass_node, end_node)
                return p1 + p2[1:]
            else: 
                return nx.shortest_path(self.G, start_node, end_node)
        except: return []

    def render_map(self, car_x, car_y, car_yaw, path, visited_nodes, path_signs, is_connected, start_node, pass_node, end_node):
        pil = self.pil_bg.copy()
        draw = ImageDraw.Draw(pil)
        
        if path:
            for i in range(len(path)-1):
                n1, n2 = path[i], path[i+1]
                color = THEME["danger"] if (n1 in visited_nodes and n2 in visited_nodes) else THEME["accent"]
                p1 = self.node_pixels.get(n1)
                p2 = self.node_pixels.get(n2)
                if p1 and p2: draw.line([p1, p2], fill=color, width=4)
        
        try: font = ImageFont.truetype("seguiemj.ttf", 20) 
        except: font = ImageFont.load_default()
        
        path_nodes = set(path)
        for s in self.signs:
            p = self.node_pixels.get(s['node'])
            if not p: continue
            s_type = s['type']
            emoji = SIGN_MAP.get(s_type, {"emoji": "?"})['emoji']
            outline = None
            
            if s['node'] in path_nodes:
                status = "⏳ PENDING"
                for ps in path_signs:
                    if ps['node'] == s['node']: 
                        status = ps.get('status', '⏳ PENDING')
                        break
                        
                if status == "✅ CONFIRMED": outline = THEME["danger"]       
                elif "🔴" in status or "🟢" in status: outline = "#00ffff"        
                else: outline = THEME["success"]                             
            
            if outline: draw.ellipse([p[0]-14, p[1]-14, p[0]+14, p[1]+14], outline=outline, width=3)
            try: draw.text((p[0]-10, p[1]-10), emoji, font=font, fill="white", embedded_color=True)
            except: draw.text((p[0]-10, p[1]-10), emoji, font=font, fill="white")
            
        def mark(n, c): 
            if n and n in self.node_pixels:
                p = self.node_pixels[n]
                draw.ellipse([p[0]-6, p[1]-6, p[0]+6, p[1]+6], fill=c)
                
        mark(start_node, THEME["success"])
        mark(pass_node, "cyan")
        mark(end_node, THEME["danger"])
        
        if is_connected:
            cx, cy = self.to_pixel(car_x, car_y)
            hx = cx + math.cos(-car_yaw)*20; hy = cy + math.sin(-car_yaw)*20
            draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill="cyan", outline="white", width=2)
            draw.line([cx, cy, hx, hy], fill="white", width=2)
            
        return pil