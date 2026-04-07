# # # # # traffic_rl/visualize.py
# # # # import pygame
# # # # import numpy as np

# # # # class TrafficRenderer:
# # # #     def __init__(self, width=800, height=400):
# # # #         pygame.init()
# # # #         self.width = width
# # # #         self.height = height
# # # #         self.screen = pygame.display.set_mode((self.width, self.height))
# # # #         pygame.display.set_caption("Traffic RL Visualization")
# # # #         self.clock = pygame.time.Clock()
# # # #         self.fps = 10  # Slow enough to watch the agent's decisions

# # # #         # Layout metrics matching the SVG concept
# # # #         self.road_w = 80
# # # #         self.cx_left = 250   # Center X of left intersection
# # # #         self.cx_right = 550  # Center X of right intersection
# # # #         self.cy = 200        # Center Y of the main E-W road

# # # #     def draw(self, env):
# # # #         for event in pygame.event.get():
# # # #             if event.type == pygame.QUIT:
# # # #                 self.close()
# # # #                 exit()
# # # #         self.screen.fill((40, 40, 40))  # Dark background

# # # #         # Draw Roads
# # # #         grey = (100, 100, 100)
# # # #         # Main E-W road
# # # #         pygame.draw.rect(self.screen, grey, (0, self.cy - self.road_w//2, self.width, self.road_w))
# # # #         # Left N-S road
# # # #         pygame.draw.rect(self.screen, grey, (self.cx_left - self.road_w//2, 0, self.road_w, self.height))
# # # #         # Right N-S road
# # # #         pygame.draw.rect(self.screen, grey, (self.cx_right - self.road_w//2, 0, self.road_w, self.height))

# # # #         # Draw details for both intersections
# # # #         self._draw_intersection(env, 0, self.cx_left, self.cy)
# # # #         self._draw_intersection(env, 1, self.cx_right, self.cy)

# # # #         pygame.display.flip()
# # # #         self.clock.tick(self.fps)

# # # #     def _draw_intersection(self, env, idx, cx, cy):
# # # #         phase = env.phases[idx] # 0 = NS Green, 1 = EW Green
# # # #         queues = env.queues[idx] # [N, S, E, W]

# # # #         # Light colors
# # # #         ns_color = (0, 255, 0) if phase == 0 else (255, 0, 0)
# # # #         ew_color = (0, 255, 0) if phase == 1 else (255, 0, 0)

# # # #         # Draw traffic lights (center indicators)
# # # #         pygame.draw.circle(self.screen, ns_color, (cx, cy - 15), 8) # NS indicator
# # # #         pygame.draw.circle(self.screen, ew_color, (cx + 15, cy), 8) # EW indicator

# # # #         # Draw queues (representing cars as small rectangles)
# # # #         car_color = (200, 200, 50)
# # # #         car_size = 10
# # # #         gap = 12

# # # #         # North Queue (waiting to go South)
# # # #         for i in range(queues[0]):
# # # #             pygame.draw.rect(self.screen, car_color, (cx - 20, cy - 50 - (i*gap), car_size, car_size))
# # # #         # South Queue (waiting to go North)
# # # #         for i in range(queues[1]):
# # # #             pygame.draw.rect(self.screen, car_color, (cx + 10, cy + 40 + (i*gap), car_size, car_size))
# # # #         # East Queue (waiting to go West)
# # # #         for i in range(queues[2]):
# # # #             pygame.draw.rect(self.screen, car_color, (cx + 40 + (i*gap), cy - 20, car_size, car_size))
# # # #         # West Queue (waiting to go East)
# # # #         for i in range(queues[3]):
# # # #             pygame.draw.rect(self.screen, car_color, (cx - 50 - (i*gap), cy + 10, car_size, car_size))

# # # #     def close(self):
# # # #         pygame.quit()

# # # import pygame
# # # import numpy as np

# # # class TrafficRenderer:
# # #     def __init__(self, width=1000, height=450):
# # #         pygame.init()
# # #         pygame.font.init()
# # #         self.width = width
# # #         self.height = height
# # #         self.screen = pygame.display.set_mode((self.width, self.height))
# # #         pygame.display.set_caption("Traffic RL - Upgraded View")
# # #         self.clock = pygame.time.Clock()
        
# # #         # SLOWED DOWN drastically so you can watch the agent think
# # #         self.fps = 2 
# # #         self.font = pygame.font.SysFont('Consolas', 20, bold=True)

# # #         self.road_w = 120
# # #         self.cx_left = 300
# # #         self.cx_right = 700
# # #         self.cy = 225

# # #     def draw(self, env):
# # #         # The anti-freeze event pump
# # #         for event in pygame.event.get():
# # #             if event.type == pygame.QUIT:
# # #                 self.close()
# # #                 exit()

# # #         self.screen.fill((30, 30, 30))

# # #         # Draw Roads
# # #         grey = (70, 70, 70)
# # #         pygame.draw.rect(self.screen, grey, (0, self.cy - self.road_w//2, self.width, self.road_w))
# # #         pygame.draw.rect(self.screen, grey, (self.cx_left - self.road_w//2, 0, self.road_w, self.height))
# # #         pygame.draw.rect(self.screen, grey, (self.cx_right - self.road_w//2, 0, self.road_w, self.height))

# # #         # Draw Intersections
# # #         self._draw_intersection(env, 0, self.cx_left, self.cy)
# # #         self._draw_intersection(env, 1, self.cx_right, self.cy)

# # #         # Overlay stats
# # #         info_text = f"Step: {env.step_count} | Mean Queue: {env.queues.mean():.2f}"
# # #         text_surface = self.font.render(info_text, True, (255, 255, 255))
# # #         self.screen.blit(text_surface, (15, 15))

# # #         pygame.display.flip()
# # #         self.clock.tick(self.fps)

# # #     def _draw_intersection(self, env, idx, cx, cy):
# # #         phase = env.phases[idx]
# # #         queues = env.queues[idx]

# # #         green = (40, 255, 40)
# # #         red = (255, 40, 40)
# # #         car_color = (255, 200, 0)

# # #         # Draw massive stop lines to indicate active phase
# # #         ns_color = green if phase == 0 else red
# # #         pygame.draw.line(self.screen, ns_color, (cx - self.road_w//2, cy - self.road_w//2), (cx + self.road_w//2, cy - self.road_w//2), 8) # Top
# # #         pygame.draw.line(self.screen, ns_color, (cx - self.road_w//2, cy + self.road_w//2), (cx + self.road_w//2, cy + self.road_w//2), 8) # Bottom

# # #         ew_color = green if phase == 1 else red
# # #         pygame.draw.line(self.screen, ew_color, (cx - self.road_w//2, cy - self.road_w//2), (cx - self.road_w//2, cy + self.road_w//2), 8) # Left
# # #         pygame.draw.line(self.screen, ew_color, (cx + self.road_w//2, cy - self.road_w//2), (cx + self.road_w//2, cy + self.road_w//2), 8) # Right

# # #         # Car dimensions
# # #         car_w, car_l = 18, 30
# # #         gap = 35

# # #         # North Queue (facing down)
# # #         for i in range(queues[0]):
# # #             pygame.draw.rect(self.screen, car_color, (cx - 35, cy - 65 - (i*gap), car_w, car_l))
# # #         # South Queue (facing up)
# # #         for i in range(queues[1]):
# # #             pygame.draw.rect(self.screen, car_color, (cx + 15, cy + 65 + (i*gap), car_w, car_l))
# # #         # East Queue (facing left)
# # #         for i in range(queues[2]):
# # #             pygame.draw.rect(self.screen, car_color, (cx + 65 + (i*gap), cy - 35, car_l, car_w))
# # #         # West Queue (facing right)
# # #         for i in range(queues[3]):
# # #             pygame.draw.rect(self.screen, car_color, (cx - 95 - (i*gap), cy + 15, car_l, car_w))

# # #     def close(self):
# # #         pygame.quit()

# # import pygame
# # import numpy as np

# # # ── Visual Constants ────────────────────────────────────────────────────────
# # SCREEN_WIDTH  = 1000
# # SCREEN_HEIGHT = 600
# # ROAD_WIDTH    = 80
# # CAR_SIZE      = 12
# # FPS           = 1  # Adjust this to speed up/slow down the "movie"

# # # Colors
# # COLOR_ROAD    = (50, 50, 50)
# # COLOR_GRASS   = (34, 139, 34)
# # COLOR_CAR     = (200, 200, 200)
# # COLOR_GREEN   = (0, 255, 0)
# # COLOR_RED     = (255, 0, 0)
# # COLOR_TEXT    = (255, 255, 255)

# # class TrafficRenderer:
# #     def __init__(self):
# #         pygame.init()
# #         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# #         pygame.display.set_caption("Traffic RL: Two-Intersection Sync")
# #         self.clock = pygame.time.Clock()
# #         self.font = pygame.font.SysFont("Arial", 18)

# #     def draw(self, env):
# #         """Draws the current state of the environment."""
# #         self.screen.fill(COLOR_GRASS)
        
# #         # 1. Draw Roads (Horizontal and two Vertical)
# #         # Main E-W Road
# #         pygame.draw.rect(self.screen, COLOR_ROAD, (0, SCREEN_HEIGHT//2 - ROAD_WIDTH//2, SCREEN_WIDTH, ROAD_WIDTH))
        
# #         # Two N-S Roads (Intersection Left at 25%, Right at 75%)
# #         il_x = SCREEN_WIDTH // 4
# #         ir_x = (SCREEN_WIDTH // 4) * 3
# #         pygame.draw.rect(self.screen, COLOR_ROAD, (il_x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
# #         pygame.draw.rect(self.screen, COLOR_ROAD, (ir_x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))

# #         # 2. Draw Intersections and Lights
# #         intersections = [il_x, ir_x]
# #         for i, x in enumerate(intersections):
# #             phase = env.phases[i]
# #             # NS Lights
# #             ns_color = COLOR_GREEN if phase == 0 else COLOR_RED
# #             pygame.draw.circle(self.screen, ns_color, (x, SCREEN_HEIGHT//2 - ROAD_WIDTH), 10) # North
# #             pygame.draw.circle(self.screen, ns_color, (x, SCREEN_HEIGHT//2 + ROAD_WIDTH), 10) # South
            
# #             # EW Lights
# #             ew_color = COLOR_GREEN if phase == 1 else COLOR_RED
# #             pygame.draw.circle(self.screen, ew_color, (x - ROAD_WIDTH, SCREEN_HEIGHT//2), 10) # West
# #             pygame.draw.circle(self.screen, ew_color, (x + ROAD_WIDTH, SCREEN_HEIGHT//2), 10) # East

# #             # 3. Draw Queues (Approaches)
# #             # Arm indices: 0=N, 1=S, 2=E, 3=W
# #             q = env.queues[i]
# #             self._draw_queue(q[0], x, SCREEN_HEIGHT//2 - ROAD_WIDTH//2, "N")
# #             self._draw_queue(q[1], x, SCREEN_HEIGHT//2 + ROAD_WIDTH//2, "S")
# #             self._draw_queue(q[2], x + ROAD_WIDTH//2, SCREEN_HEIGHT//2, "E")
# #             self._draw_queue(q[3], x - ROAD_WIDTH//2, SCREEN_HEIGHT//2, "W")

# #         # 4. Draw Corridor Traffic (The "Sync" visualization)
# #         # Visualize the deque as moving blocks between the two intersections
# #         corridor_width = ir_x - il_x - ROAD_WIDTH
# #         step_size = corridor_width / len(env.corridor_0to1)
        
# #         # IL to IR (0 -> 1)
# #         for idx, count in enumerate(env.corridor_0to1):
# #             if count > 0:
# #                 pos_x = il_x + ROAD_WIDTH//2 + (idx * step_size) + step_size//2
# #                 pygame.draw.rect(self.screen, (0, 150, 255), (pos_x, SCREEN_HEIGHT//2 - 10, CAR_SIZE, CAR_SIZE))
        
# #         # IR to IL (1 -> 0)
# #         for idx, count in enumerate(reversed(list(env.corridor_1to0))):
# #             if count > 0:
# #                 pos_x = il_x + ROAD_WIDTH//2 + (idx * step_size) + step_size//2
# #                 pygame.draw.rect(self.screen, (255, 150, 0), (pos_x, SCREEN_HEIGHT//2 + 10, CAR_SIZE, CAR_SIZE))

# #         # 5. UI / Stats
# #         reward_text = self.font.render(f"Step: {env.step_count} | Queued: {env.queues.sum()}", True, COLOR_TEXT)
# #         self.screen.blit(reward_text, (20, 20))

# #         pygame.display.flip()
# #         self.clock.tick(FPS)

# #     def _draw_queue(self, count, x, y, direction):
# #         """Helper to draw clusters of cars based on queue count."""
# #         for i in range(int(count)):
# #             offset = (i + 1) * (CAR_SIZE + 2)
# #             if direction == "N": pos = (x - CAR_SIZE//2, y - offset)
# #             elif direction == "S": pos = (x - CAR_SIZE//2, y + offset - CAR_SIZE)
# #             elif direction == "E": pos = (x + offset - CAR_SIZE, y - CAR_SIZE//2)
# #             elif direction == "W": pos = (x - offset, y - CAR_SIZE//2)
            
# #             pygame.draw.rect(self.screen, COLOR_CAR, (pos[0], pos[1], CAR_SIZE, CAR_SIZE))

# #     def close(self):
# #         pygame.quit()

# import pygame
# import numpy as np

# # ── Visual Constants ────────────────────────────────────────────────────────
# SCREEN_WIDTH  = 1000
# SCREEN_HEIGHT = 600
# ROAD_WIDTH    = 100  # Wider roads for dual lanes
# CAR_WIDTH     = 20
# CAR_HEIGHT    = 12
# FPS           = 1    # Slowed down as requested

# # Colors
# COLOR_ROAD    = (40, 40, 40)
# COLOR_GRASS   = (30, 120, 30)
# COLOR_MARKING = (200, 200, 200) # Lane dividers
# COLOR_TEXT    = (255, 255, 255)

# # Car Colors by Direction
# COLORS = {
#     "N": (255, 80, 80),   # Red-ish
#     "S": (80, 255, 80),   # Green-ish
#     "E": (80, 80, 255),   # Blue-ish
#     "W": (255, 255, 80)   # Yellow-ish
# }

# class TrafficRenderer:
#     def __init__(self):
#         pygame.init()
#         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#         pygame.display.set_caption("Traffic RL: Two-Intersection Sync")
#         self.clock = pygame.time.Clock()
#         self.font = pygame.font.SysFont("Consolas", 20, bold=True)

#     def _draw_car(self, x, y, direction):
#         """Draws a car with headlights to show direction."""
#         color = COLORS.get(direction, (200, 200, 200))
        
#         # Determine orientation
#         if direction in ["E", "W"]:
#             w, h = CAR_WIDTH, CAR_HEIGHT
#         else:
#             w, h = CAR_HEIGHT, CAR_WIDTH
            
#         rect = pygame.Rect(x - w//2, y - h//2, w, h)
#         pygame.draw.rect(self.screen, color, rect, border_radius=3)
#         pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=3) # Outline

#         # Small headlights
#         headlight_color = (255, 255, 200)
#         if direction == "E":
#             pygame.draw.circle(self.screen, headlight_color, (x + w//2, y - h//4), 2)
#             pygame.draw.circle(self.screen, headlight_color, (x + w//2, y + h//4), 2)
#         elif direction == "W":
#             pygame.draw.circle(self.screen, headlight_color, (x - w//2, y - h//4), 2)
#             pygame.draw.circle(self.screen, headlight_color, (x - w//2, y + h//4), 2)
#         elif direction == "S":
#             pygame.draw.circle(self.screen, headlight_color, (x - h//4, y + w//2), 2)
#             pygame.draw.circle(self.screen, headlight_color, (x + h//4, y + w//2), 2)
#         elif direction == "N":
#             pygame.draw.circle(self.screen, headlight_color, (x - h//4, y - w//2), 2)
#             pygame.draw.circle(self.screen, headlight_color, (x + h//4, y - w//2), 2)

#     def draw(self, env):
#         self.screen.fill(COLOR_GRASS)
        
#         il_x = SCREEN_WIDTH // 4
#         ir_x = (SCREEN_WIDTH // 4) * 3
#         mid_y = SCREEN_HEIGHT // 2

#         # 1. Draw Roads & Lane Markings
#         pygame.draw.rect(self.screen, COLOR_ROAD, (0, mid_y - ROAD_WIDTH//2, SCREEN_WIDTH, ROAD_WIDTH))
#         pygame.draw.line(self.screen, COLOR_MARKING, (0, mid_y), (SCREEN_WIDTH, mid_y), 2) # E-W Divider

#         for x in [il_x, ir_x]:
#             pygame.draw.rect(self.screen, COLOR_ROAD, (x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
#             pygame.draw.line(self.screen, COLOR_MARKING, (x, 0), (x, SCREEN_HEIGHT), 2) # N-S Divider

#         # 2. Draw Queued Cars (Separated into proper lanes)
#         for i, x_center in enumerate([il_x, ir_x]):
#             q = env.queues[i]
#             lane_offset = ROAD_WIDTH // 4
            
#             # North Arm (Driving South, Right side of N-S road)
#             for k in range(int(q[0])):
#                 self._draw_car(x_center + lane_offset, mid_y - ROAD_WIDTH//2 - 20 - (k*25), "S")
#             # South Arm (Driving North, Left side of N-S road)
#             for k in range(int(q[1])):
#                 self._draw_car(x_center - lane_offset, mid_y + ROAD_WIDTH//2 + 20 + (k*25), "N")
#             # East Arm (Driving West, Top side of E-W road)
#             for k in range(int(q[2])):
#                 self._draw_car(x_center + ROAD_WIDTH//2 + 20 + (k*25), mid_y - lane_offset, "W")
#             # West Arm (Driving East, Bottom side of E-W road)
#             for k in range(int(q[3])):
#                 self._draw_car(x_center - ROAD_WIDTH//2 - 20 - (k*25), mid_y + lane_offset, "E")

#             # 3. Draw Traffic Lights
#             p = env.phases[i]
#             # NS Lights
#             ns_c = (0, 255, 0) if p == 0 else (255, 0, 0)
#             pygame.draw.circle(self.screen, ns_c, (x_center, mid_y - ROAD_WIDTH//2 - 10), 8)
#             # EW Lights
#             ew_c = (0, 255, 0) if p == 1 else (255, 0, 0)
#             pygame.draw.circle(self.screen, ew_c, (x_center - ROAD_WIDTH//2 - 10, mid_y), 8)

#         # 4. Draw Corridor (The Sync Area)
#         corr_len = ir_x - il_x - ROAD_WIDTH
#         step_w = corr_len / len(env.corridor_0to1)
#         lane_offset = ROAD_WIDTH // 4

#         for idx, count in enumerate(env.corridor_0to1):
#             if count > 0:
#                 cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
#                 self._draw_car(cx, mid_y + lane_offset, "E")
        
#         for idx, count in enumerate(reversed(list(env.corridor_1to0))):
#             if count > 0:
#                 cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
#                 self._draw_car(cx, mid_y - lane_offset, "W")

#         # 5. UI
#         info = self.font.render(f"STEP: {env.step_count} | TOTAL QUEUED: {int(env.queues.sum())}", True, COLOR_TEXT)
#         self.screen.blit(info, (20, 20))

#         pygame.display.flip()
#         self.clock.tick(FPS)

#     def close(self):
#         pygame.quit()

import pygame
import numpy as np

# ── Visual Constants ────────────────────────────────────────────────────────
SCREEN_WIDTH  = 1000
SCREEN_HEIGHT = 600
ROAD_WIDTH    = 100  
CAR_WIDTH     = 20
CAR_HEIGHT    = 12
FPS           = 1   # Slightly faster for smoother motion

# Colors
COLOR_ROAD    = (40, 40, 40)
COLOR_GRASS   = (30, 120, 30)
COLOR_MARKING = (200, 200, 200) 
COLOR_TEXT    = (255, 255, 255)

# Car Colors by Direction
COLORS = {
    "N": (255, 80, 80),   # Red-ish
    "S": (80, 255, 80),   # Green-ish
    "E": (80, 80, 255),   # Blue-ish
    "W": (255, 255, 80)   # Yellow-ish
}
COLOR_EV = (255, 40, 40)       # Bright red for emergency vehicles
COLOR_EV_SIREN = (40, 100, 255) # Blue siren accent

class TrafficRenderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Traffic RL: Two-Intersection Sync")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)

    def _draw_car(self, x, y, direction, is_ev=False):
        """Draws a car with headlights to show direction."""
        if is_ev:
            color = COLOR_EV
        else:
            color = COLORS.get(direction, (200, 200, 200))
        w, h = (CAR_WIDTH, CAR_HEIGHT) if direction in ["E", "W"] else (CAR_HEIGHT, CAR_WIDTH)
            
        rect = pygame.Rect(x - w//2, y - h//2, w, h)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        if is_ev:
            # Thick white border + siren light to make EVs visually distinct
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=3)
            # Blue siren on top
            siren_x, siren_y = x, y
            pygame.draw.circle(self.screen, COLOR_EV_SIREN, (int(siren_x), int(siren_y)), 3)
        else:
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=3) 

        # Headlights
        headlight_color = (255, 255, 200)
        offsets = {
            "E": [(w//2, -h//4), (w//2, h//4)],
            "W": [(-w//2, -h//4), (-w//2, h//4)],
            "S": [(-h//4, w//2), (h//4, w//2)],
            "N": [(-h//4, -w//2), (h//4, -w//2)]
        }
        for dx, dy in offsets[direction]:
            pygame.draw.circle(self.screen, headlight_color, (int(x + dx), int(y + dy)), 2)

    def draw(self, env):
        self.screen.fill(COLOR_GRASS)
        il_x = SCREEN_WIDTH // 4
        ir_x = (SCREEN_WIDTH // 4) * 3
        mid_y = SCREEN_HEIGHT // 2
        lane_offset = ROAD_WIDTH // 4

        # 1. Draw Continuous Roads
        pygame.draw.rect(self.screen, COLOR_ROAD, (0, mid_y - ROAD_WIDTH//2, SCREEN_WIDTH, ROAD_WIDTH))
        pygame.draw.line(self.screen, COLOR_MARKING, (0, mid_y), (SCREEN_WIDTH, mid_y), 2)

        for x in [il_x, ir_x]:
            pygame.draw.rect(self.screen, COLOR_ROAD, (x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
            pygame.draw.line(self.screen, COLOR_MARKING, (x, 0), (x, SCREEN_HEIGHT), 2)

        # 2. Draw Queued & "Approaching" Cars (standard + EV)
        has_ev = hasattr(env, 'ev_queues') and env.ev_queues is not None
        for i, x_center in enumerate([il_x, ir_x]):
            q = env.queues[i]
            eq = env.ev_queues[i] if has_ev else [0, 0, 0, 0]

            # North Arm (S-bound) - Entering from top edge
            pos = 0
            for k in range(int(q[0])):
                self._draw_car(x_center + lane_offset, (mid_y - ROAD_WIDTH//2 - 20) - (pos*25), "S")
                pos += 1
            for k in range(int(eq[0])):
                self._draw_car(x_center + lane_offset, (mid_y - ROAD_WIDTH//2 - 20) - (pos*25), "S", is_ev=True)
                pos += 1

            # South Arm (N-bound) - Entering from bottom edge
            pos = 0
            for k in range(int(q[1])):
                self._draw_car(x_center - lane_offset, (mid_y + ROAD_WIDTH//2 + 20) + (pos*25), "N")
                pos += 1
            for k in range(int(eq[1])):
                self._draw_car(x_center - lane_offset, (mid_y + ROAD_WIDTH//2 + 20) + (pos*25), "N", is_ev=True)
                pos += 1

            # East Arm (W-bound) - IR enters from right edge; IL gets from corridor
            pos = 0
            if i == 1: # Intersection Right
                for k in range(int(q[2])):
                    self._draw_car(SCREEN_WIDTH - 20 - (pos*25), mid_y - lane_offset, "W")
                    pos += 1
                for k in range(int(eq[2])):
                    self._draw_car(SCREEN_WIDTH - 20 - (pos*25), mid_y - lane_offset, "W", is_ev=True)
                    pos += 1
            else: # Intersection Left
                for k in range(int(q[2])):
                    self._draw_car(x_center + ROAD_WIDTH//2 + 20 + (pos*25), mid_y - lane_offset, "W")
                    pos += 1
                for k in range(int(eq[2])):
                    self._draw_car(x_center + ROAD_WIDTH//2 + 20 + (pos*25), mid_y - lane_offset, "W", is_ev=True)
                    pos += 1

            # West Arm (E-bound) - IL enters from left edge; IR gets from corridor
            pos = 0
            if i == 0: # Intersection Left
                for k in range(int(q[3])):
                    self._draw_car(20 + (pos*25), mid_y + lane_offset, "E")
                    pos += 1
                for k in range(int(eq[3])):
                    self._draw_car(20 + (pos*25), mid_y + lane_offset, "E", is_ev=True)
                    pos += 1
            else: # Intersection Right
                for k in range(int(q[3])):
                    self._draw_car(x_center - ROAD_WIDTH//2 - 20 - (pos*25), mid_y + lane_offset, "E")
                    pos += 1
                for k in range(int(eq[3])):
                    self._draw_car(x_center - ROAD_WIDTH//2 - 20 - (pos*25), mid_y + lane_offset, "E", is_ev=True)
                    pos += 1

            # 3. Traffic Lights
            p = env.phases[i]
            ns_c = (0, 255, 0) if p == 0 else (255, 0, 0)
            ew_c = (0, 255, 0) if p == 1 else (255, 0, 0)
            pygame.draw.circle(self.screen, ns_c, (x_center, mid_y - ROAD_WIDTH//2 - 15), 10)
            pygame.draw.circle(self.screen, ew_c, (x_center - ROAD_WIDTH//2 - 15, mid_y), 10)

        # 4. Draw Corridor (The Sync Area)
        corr_len = ir_x - il_x - ROAD_WIDTH
        step_w = corr_len / len(env.corridor_0to1)
        for idx, count in enumerate(env.corridor_0to1):
            if count > 0:
                cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
                self._draw_car(cx, mid_y + lane_offset, "E")
        for idx, count in enumerate(reversed(list(env.corridor_1to0))):
            if count > 0:
                cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
                self._draw_car(cx, mid_y - lane_offset, "W")

        # 5. UI Stats
        ev_total = int(env.ev_queues.sum()) if has_ev else 0
        info = self.font.render(
            f"STEP: {env.step_count} | STD QUEUED: {int(env.queues.sum())} | EV QUEUED: {ev_total}",
            True, COLOR_TEXT
        )
        self.screen.blit(info, (20, 20))
        if ev_total > 0:
            ev_warn = self.font.render("⚠ EMERGENCY VEHICLES WAITING", True, COLOR_EV)
            self.screen.blit(ev_warn, (20, 50))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # ──────────────────────────────────────────────────────────────────────────
    # Mock Demo Mode
    # Use this to verify the UI and EV rendering without running a full simulation
    # ──────────────────────────────────────────────────────────────────────────
    import numpy as np
    print("Starting Traffic RL Visualization Demo...")
    print("Close the window or press Ctrl+C to exit.")

    class MockEnv:
        def __init__(self):
            self.queues = np.zeros((2, 4), dtype=np.int32)
            self.ev_queues = np.zeros((2, 4), dtype=np.int32)
            self.phases = np.zeros(2, dtype=np.int32)
            self.step_count = 0
            self.corridor_0to1 = [0, 0, 0, 0, 0]
            self.corridor_1to0 = [0, 0, 0, 0, 0]

    renderer = TrafficRenderer()
    env = MockEnv()
    
    try:
        running = True
        while running:
            env.step_count += 1
            
            # Randomly update queues and phases for the demo
            if env.step_count % 5 == 0:
                env.phases = np.random.randint(0, 2, 2)
                env.queues = np.random.randint(0, 8, (2, 4))
                # 20% chance of an EV appearing in any arm
                env.ev_queues = (np.random.random((2, 4)) < 0.2).astype(np.int32)
            
            renderer.draw(env)
            # Add a small event loop to handle window closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        renderer.close()