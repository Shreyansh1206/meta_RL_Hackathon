# # # # # # # traffic_rl/visualize.py
# # # # # # import pygame
# # # # # # import numpy as np

# # # # # # class TrafficRenderer:
# # # # # #     def __init__(self, width=800, height=400):
# # # # # #         pygame.init()
# # # # # #         self.width = width
# # # # # #         self.height = height
# # # # # #         self.screen = pygame.display.set_mode((self.width, self.height))
# # # # # #         pygame.display.set_caption("Traffic RL Visualization")
# # # # # #         self.clock = pygame.time.Clock()
# # # # # #         self.fps = 10  # Slow enough to watch the agent's decisions

# # # # # #         # Layout metrics matching the SVG concept
# # # # # #         self.road_w = 80
# # # # # #         self.cx_left = 250   # Center X of left intersection
# # # # # #         self.cx_right = 550  # Center X of right intersection
# # # # # #         self.cy = 200        # Center Y of the main E-W road

# # # # # #     def draw(self, env):
# # # # # #         for event in pygame.event.get():
# # # # # #             if event.type == pygame.QUIT:
# # # # # #                 self.close()
# # # # # #                 exit()
# # # # # #         self.screen.fill((40, 40, 40))  # Dark background

# # # # # #         # Draw Roads
# # # # # #         grey = (100, 100, 100)
# # # # # #         # Main E-W road
# # # # # #         pygame.draw.rect(self.screen, grey, (0, self.cy - self.road_w//2, self.width, self.road_w))
# # # # # #         # Left N-S road
# # # # # #         pygame.draw.rect(self.screen, grey, (self.cx_left - self.road_w//2, 0, self.road_w, self.height))
# # # # # #         # Right N-S road
# # # # # #         pygame.draw.rect(self.screen, grey, (self.cx_right - self.road_w//2, 0, self.road_w, self.height))

# # # # # #         # Draw details for both intersections
# # # # # #         self._draw_intersection(env, 0, self.cx_left, self.cy)
# # # # # #         self._draw_intersection(env, 1, self.cx_right, self.cy)

# # # # # #         pygame.display.flip()
# # # # # #         self.clock.tick(self.fps)

# # # # # #     def _draw_intersection(self, env, idx, cx, cy):
# # # # # #         phase = env.phases[idx] # 0 = NS Green, 1 = EW Green
# # # # # #         queues = env.queues[idx] # [N, S, E, W]

# # # # # #         # Light colors
# # # # # #         ns_color = (0, 255, 0) if phase == 0 else (255, 0, 0)
# # # # # #         ew_color = (0, 255, 0) if phase == 1 else (255, 0, 0)

# # # # # #         # Draw traffic lights (center indicators)
# # # # # #         pygame.draw.circle(self.screen, ns_color, (cx, cy - 15), 8) # NS indicator
# # # # # #         pygame.draw.circle(self.screen, ew_color, (cx + 15, cy), 8) # EW indicator

# # # # # #         # Draw queues (representing cars as small rectangles)
# # # # # #         car_color = (200, 200, 50)
# # # # # #         car_size = 10
# # # # # #         gap = 12

# # # # # #         # North Queue (waiting to go South)
# # # # # #         for i in range(queues[0]):
# # # # # #             pygame.draw.rect(self.screen, car_color, (cx - 20, cy - 50 - (i*gap), car_size, car_size))
# # # # # #         # South Queue (waiting to go North)
# # # # # #         for i in range(queues[1]):
# # # # # #             pygame.draw.rect(self.screen, car_color, (cx + 10, cy + 40 + (i*gap), car_size, car_size))
# # # # # #         # East Queue (waiting to go West)
# # # # # #         for i in range(queues[2]):
# # # # # #             pygame.draw.rect(self.screen, car_color, (cx + 40 + (i*gap), cy - 20, car_size, car_size))
# # # # # #         # West Queue (waiting to go East)
# # # # # #         for i in range(queues[3]):
# # # # # #             pygame.draw.rect(self.screen, car_color, (cx - 50 - (i*gap), cy + 10, car_size, car_size))

# # # # # #     def close(self):
# # # # # #         pygame.quit()

# # # # # import pygame
# # # # # import numpy as np

# # # # # class TrafficRenderer:
# # # # #     def __init__(self, width=1000, height=450):
# # # # #         pygame.init()
# # # # #         pygame.font.init()
# # # # #         self.width = width
# # # # #         self.height = height
# # # # #         self.screen = pygame.display.set_mode((self.width, self.height))
# # # # #         pygame.display.set_caption("Traffic RL - Upgraded View")
# # # # #         self.clock = pygame.time.Clock()
        
# # # # #         # SLOWED DOWN drastically so you can watch the agent think
# # # # #         self.fps = 2 
# # # # #         self.font = pygame.font.SysFont('Consolas', 20, bold=True)

# # # # #         self.road_w = 120
# # # # #         self.cx_left = 300
# # # # #         self.cx_right = 700
# # # # #         self.cy = 225

# # # # #     def draw(self, env):
# # # # #         # The anti-freeze event pump
# # # # #         for event in pygame.event.get():
# # # # #             if event.type == pygame.QUIT:
# # # # #                 self.close()
# # # # #                 exit()

# # # # #         self.screen.fill((30, 30, 30))

# # # # #         # Draw Roads
# # # # #         grey = (70, 70, 70)
# # # # #         pygame.draw.rect(self.screen, grey, (0, self.cy - self.road_w//2, self.width, self.road_w))
# # # # #         pygame.draw.rect(self.screen, grey, (self.cx_left - self.road_w//2, 0, self.road_w, self.height))
# # # # #         pygame.draw.rect(self.screen, grey, (self.cx_right - self.road_w//2, 0, self.road_w, self.height))

# # # # #         # Draw Intersections
# # # # #         self._draw_intersection(env, 0, self.cx_left, self.cy)
# # # # #         self._draw_intersection(env, 1, self.cx_right, self.cy)

# # # # #         # Overlay stats
# # # # #         info_text = f"Step: {env.step_count} | Mean Queue: {env.queues.mean():.2f}"
# # # # #         text_surface = self.font.render(info_text, True, (255, 255, 255))
# # # # #         self.screen.blit(text_surface, (15, 15))

# # # # #         pygame.display.flip()
# # # # #         self.clock.tick(self.fps)

# # # # #     def _draw_intersection(self, env, idx, cx, cy):
# # # # #         phase = env.phases[idx]
# # # # #         queues = env.queues[idx]

# # # # #         green = (40, 255, 40)
# # # # #         red = (255, 40, 40)
# # # # #         car_color = (255, 200, 0)

# # # # #         # Draw massive stop lines to indicate active phase
# # # # #         ns_color = green if phase == 0 else red
# # # # #         pygame.draw.line(self.screen, ns_color, (cx - self.road_w//2, cy - self.road_w//2), (cx + self.road_w//2, cy - self.road_w//2), 8) # Top
# # # # #         pygame.draw.line(self.screen, ns_color, (cx - self.road_w//2, cy + self.road_w//2), (cx + self.road_w//2, cy + self.road_w//2), 8) # Bottom

# # # # #         ew_color = green if phase == 1 else red
# # # # #         pygame.draw.line(self.screen, ew_color, (cx - self.road_w//2, cy - self.road_w//2), (cx - self.road_w//2, cy + self.road_w//2), 8) # Left
# # # # #         pygame.draw.line(self.screen, ew_color, (cx + self.road_w//2, cy - self.road_w//2), (cx + self.road_w//2, cy + self.road_w//2), 8) # Right

# # # # #         # Car dimensions
# # # # #         car_w, car_l = 18, 30
# # # # #         gap = 35

# # # # #         # North Queue (facing down)
# # # # #         for i in range(queues[0]):
# # # # #             pygame.draw.rect(self.screen, car_color, (cx - 35, cy - 65 - (i*gap), car_w, car_l))
# # # # #         # South Queue (facing up)
# # # # #         for i in range(queues[1]):
# # # # #             pygame.draw.rect(self.screen, car_color, (cx + 15, cy + 65 + (i*gap), car_w, car_l))
# # # # #         # East Queue (facing left)
# # # # #         for i in range(queues[2]):
# # # # #             pygame.draw.rect(self.screen, car_color, (cx + 65 + (i*gap), cy - 35, car_l, car_w))
# # # # #         # West Queue (facing right)
# # # # #         for i in range(queues[3]):
# # # # #             pygame.draw.rect(self.screen, car_color, (cx - 95 - (i*gap), cy + 15, car_l, car_w))

# # # # #     def close(self):
# # # # #         pygame.quit()

# # # # import pygame
# # # # import numpy as np

# # # # # ── Visual Constants ────────────────────────────────────────────────────────
# # # # SCREEN_WIDTH  = 1000
# # # # SCREEN_HEIGHT = 600
# # # # ROAD_WIDTH    = 80
# # # # CAR_SIZE      = 12
# # # # FPS           = 1  # Adjust this to speed up/slow down the "movie"

# # # # # Colors
# # # # COLOR_ROAD    = (50, 50, 50)
# # # # COLOR_GRASS   = (34, 139, 34)
# # # # COLOR_CAR     = (200, 200, 200)
# # # # COLOR_GREEN   = (0, 255, 0)
# # # # COLOR_RED     = (255, 0, 0)
# # # # COLOR_TEXT    = (255, 255, 255)

# # # # class TrafficRenderer:
# # # #     def __init__(self):
# # # #         pygame.init()
# # # #         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# # # #         pygame.display.set_caption("Traffic RL: Two-Intersection Sync")
# # # #         self.clock = pygame.time.Clock()
# # # #         self.font = pygame.font.SysFont("Arial", 18)

# # # #     def draw(self, env):
# # # #         """Draws the current state of the environment."""
# # # #         self.screen.fill(COLOR_GRASS)
        
# # # #         # 1. Draw Roads (Horizontal and two Vertical)
# # # #         # Main E-W Road
# # # #         pygame.draw.rect(self.screen, COLOR_ROAD, (0, SCREEN_HEIGHT//2 - ROAD_WIDTH//2, SCREEN_WIDTH, ROAD_WIDTH))
        
# # # #         # Two N-S Roads (Intersection Left at 25%, Right at 75%)
# # # #         il_x = SCREEN_WIDTH // 4
# # # #         ir_x = (SCREEN_WIDTH // 4) * 3
# # # #         pygame.draw.rect(self.screen, COLOR_ROAD, (il_x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
# # # #         pygame.draw.rect(self.screen, COLOR_ROAD, (ir_x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))

# # # #         # 2. Draw Intersections and Lights
# # # #         intersections = [il_x, ir_x]
# # # #         for i, x in enumerate(intersections):
# # # #             phase = env.phases[i]
# # # #             # NS Lights
# # # #             ns_color = COLOR_GREEN if phase == 0 else COLOR_RED
# # # #             pygame.draw.circle(self.screen, ns_color, (x, SCREEN_HEIGHT//2 - ROAD_WIDTH), 10) # North
# # # #             pygame.draw.circle(self.screen, ns_color, (x, SCREEN_HEIGHT//2 + ROAD_WIDTH), 10) # South
            
# # # #             # EW Lights
# # # #             ew_color = COLOR_GREEN if phase == 1 else COLOR_RED
# # # #             pygame.draw.circle(self.screen, ew_color, (x - ROAD_WIDTH, SCREEN_HEIGHT//2), 10) # West
# # # #             pygame.draw.circle(self.screen, ew_color, (x + ROAD_WIDTH, SCREEN_HEIGHT//2), 10) # East

# # # #             # 3. Draw Queues (Approaches)
# # # #             # Arm indices: 0=N, 1=S, 2=E, 3=W
# # # #             q = env.queues[i]
# # # #             self._draw_queue(q[0], x, SCREEN_HEIGHT//2 - ROAD_WIDTH//2, "N")
# # # #             self._draw_queue(q[1], x, SCREEN_HEIGHT//2 + ROAD_WIDTH//2, "S")
# # # #             self._draw_queue(q[2], x + ROAD_WIDTH//2, SCREEN_HEIGHT//2, "E")
# # # #             self._draw_queue(q[3], x - ROAD_WIDTH//2, SCREEN_HEIGHT//2, "W")

# # # #         # 4. Draw Corridor Traffic (The "Sync" visualization)
# # # #         # Visualize the deque as moving blocks between the two intersections
# # # #         corridor_width = ir_x - il_x - ROAD_WIDTH
# # # #         step_size = corridor_width / len(env.corridor_0to1)
        
# # # #         # IL to IR (0 -> 1)
# # # #         for idx, count in enumerate(env.corridor_0to1):
# # # #             if count > 0:
# # # #                 pos_x = il_x + ROAD_WIDTH//2 + (idx * step_size) + step_size//2
# # # #                 pygame.draw.rect(self.screen, (0, 150, 255), (pos_x, SCREEN_HEIGHT//2 - 10, CAR_SIZE, CAR_SIZE))
        
# # # #         # IR to IL (1 -> 0)
# # # #         for idx, count in enumerate(reversed(list(env.corridor_1to0))):
# # # #             if count > 0:
# # # #                 pos_x = il_x + ROAD_WIDTH//2 + (idx * step_size) + step_size//2
# # # #                 pygame.draw.rect(self.screen, (255, 150, 0), (pos_x, SCREEN_HEIGHT//2 + 10, CAR_SIZE, CAR_SIZE))

# # # #         # 5. UI / Stats
# # # #         reward_text = self.font.render(f"Step: {env.step_count} | Queued: {env.queues.sum()}", True, COLOR_TEXT)
# # # #         self.screen.blit(reward_text, (20, 20))

# # # #         pygame.display.flip()
# # # #         self.clock.tick(FPS)

# # # #     def _draw_queue(self, count, x, y, direction):
# # # #         """Helper to draw clusters of cars based on queue count."""
# # # #         for i in range(int(count)):
# # # #             offset = (i + 1) * (CAR_SIZE + 2)
# # # #             if direction == "N": pos = (x - CAR_SIZE//2, y - offset)
# # # #             elif direction == "S": pos = (x - CAR_SIZE//2, y + offset - CAR_SIZE)
# # # #             elif direction == "E": pos = (x + offset - CAR_SIZE, y - CAR_SIZE//2)
# # # #             elif direction == "W": pos = (x - offset, y - CAR_SIZE//2)
            
# # # #             pygame.draw.rect(self.screen, COLOR_CAR, (pos[0], pos[1], CAR_SIZE, CAR_SIZE))

# # # #     def close(self):
# # # #         pygame.quit()

# # # import pygame
# # # import numpy as np

# # # # ── Visual Constants ────────────────────────────────────────────────────────
# # # SCREEN_WIDTH  = 1000
# # # SCREEN_HEIGHT = 600
# # # ROAD_WIDTH    = 100  # Wider roads for dual lanes
# # # CAR_WIDTH     = 20
# # # CAR_HEIGHT    = 12
# # # FPS           = 1    # Slowed down as requested

# # # # Colors
# # # COLOR_ROAD    = (40, 40, 40)
# # # COLOR_GRASS   = (30, 120, 30)
# # # COLOR_MARKING = (200, 200, 200) # Lane dividers
# # # COLOR_TEXT    = (255, 255, 255)

# # # # Car Colors by Direction
# # # COLORS = {
# # #     "N": (255, 80, 80),   # Red-ish
# # #     "S": (80, 255, 80),   # Green-ish
# # #     "E": (80, 80, 255),   # Blue-ish
# # #     "W": (255, 255, 80)   # Yellow-ish
# # # }

# # # class TrafficRenderer:
# # #     def __init__(self):
# # #         pygame.init()
# # #         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# # #         pygame.display.set_caption("Traffic RL: Two-Intersection Sync")
# # #         self.clock = pygame.time.Clock()
# # #         self.font = pygame.font.SysFont("Consolas", 20, bold=True)

# # #     def _draw_car(self, x, y, direction):
# # #         """Draws a car with headlights to show direction."""
# # #         color = COLORS.get(direction, (200, 200, 200))
        
# # #         # Determine orientation
# # #         if direction in ["E", "W"]:
# # #             w, h = CAR_WIDTH, CAR_HEIGHT
# # #         else:
# # #             w, h = CAR_HEIGHT, CAR_WIDTH
            
# # #         rect = pygame.Rect(x - w//2, y - h//2, w, h)
# # #         pygame.draw.rect(self.screen, color, rect, border_radius=3)
# # #         pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=3) # Outline

# # #         # Small headlights
# # #         headlight_color = (255, 255, 200)
# # #         if direction == "E":
# # #             pygame.draw.circle(self.screen, headlight_color, (x + w//2, y - h//4), 2)
# # #             pygame.draw.circle(self.screen, headlight_color, (x + w//2, y + h//4), 2)
# # #         elif direction == "W":
# # #             pygame.draw.circle(self.screen, headlight_color, (x - w//2, y - h//4), 2)
# # #             pygame.draw.circle(self.screen, headlight_color, (x - w//2, y + h//4), 2)
# # #         elif direction == "S":
# # #             pygame.draw.circle(self.screen, headlight_color, (x - h//4, y + w//2), 2)
# # #             pygame.draw.circle(self.screen, headlight_color, (x + h//4, y + w//2), 2)
# # #         elif direction == "N":
# # #             pygame.draw.circle(self.screen, headlight_color, (x - h//4, y - w//2), 2)
# # #             pygame.draw.circle(self.screen, headlight_color, (x + h//4, y - w//2), 2)

# # #     def draw(self, env):
# # #         self.screen.fill(COLOR_GRASS)
        
# # #         il_x = SCREEN_WIDTH // 4
# # #         ir_x = (SCREEN_WIDTH // 4) * 3
# # #         mid_y = SCREEN_HEIGHT // 2

# # #         # 1. Draw Roads & Lane Markings
# # #         pygame.draw.rect(self.screen, COLOR_ROAD, (0, mid_y - ROAD_WIDTH//2, SCREEN_WIDTH, ROAD_WIDTH))
# # #         pygame.draw.line(self.screen, COLOR_MARKING, (0, mid_y), (SCREEN_WIDTH, mid_y), 2) # E-W Divider

# # #         for x in [il_x, ir_x]:
# # #             pygame.draw.rect(self.screen, COLOR_ROAD, (x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
# # #             pygame.draw.line(self.screen, COLOR_MARKING, (x, 0), (x, SCREEN_HEIGHT), 2) # N-S Divider

# # #         # 2. Draw Queued Cars (Separated into proper lanes)
# # #         for i, x_center in enumerate([il_x, ir_x]):
# # #             q = env.queues[i]
# # #             lane_offset = ROAD_WIDTH // 4
            
# # #             # North Arm (Driving South, Right side of N-S road)
# # #             for k in range(int(q[0])):
# # #                 self._draw_car(x_center + lane_offset, mid_y - ROAD_WIDTH//2 - 20 - (k*25), "S")
# # #             # South Arm (Driving North, Left side of N-S road)
# # #             for k in range(int(q[1])):
# # #                 self._draw_car(x_center - lane_offset, mid_y + ROAD_WIDTH//2 + 20 + (k*25), "N")
# # #             # East Arm (Driving West, Top side of E-W road)
# # #             for k in range(int(q[2])):
# # #                 self._draw_car(x_center + ROAD_WIDTH//2 + 20 + (k*25), mid_y - lane_offset, "W")
# # #             # West Arm (Driving East, Bottom side of E-W road)
# # #             for k in range(int(q[3])):
# # #                 self._draw_car(x_center - ROAD_WIDTH//2 - 20 - (k*25), mid_y + lane_offset, "E")

# # #             # 3. Draw Traffic Lights
# # #             p = env.phases[i]
# # #             # NS Lights
# # #             ns_c = (0, 255, 0) if p == 0 else (255, 0, 0)
# # #             pygame.draw.circle(self.screen, ns_c, (x_center, mid_y - ROAD_WIDTH//2 - 10), 8)
# # #             # EW Lights
# # #             ew_c = (0, 255, 0) if p == 1 else (255, 0, 0)
# # #             pygame.draw.circle(self.screen, ew_c, (x_center - ROAD_WIDTH//2 - 10, mid_y), 8)

# # #         # 4. Draw Corridor (The Sync Area)
# # #         corr_len = ir_x - il_x - ROAD_WIDTH
# # #         step_w = corr_len / len(env.corridor_0to1)
# # #         lane_offset = ROAD_WIDTH // 4

# # #         for idx, count in enumerate(env.corridor_0to1):
# # #             if count > 0:
# # #                 cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
# # #                 self._draw_car(cx, mid_y + lane_offset, "E")
        
# # #         for idx, count in enumerate(reversed(list(env.corridor_1to0))):
# # #             if count > 0:
# # #                 cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
# # #                 self._draw_car(cx, mid_y - lane_offset, "W")

# # #         # 5. UI
# # #         info = self.font.render(f"STEP: {env.step_count} | TOTAL QUEUED: {int(env.queues.sum())}", True, COLOR_TEXT)
# # #         self.screen.blit(info, (20, 20))

# # #         pygame.display.flip()
# # #         self.clock.tick(FPS)

# # #     def close(self):
# # #         pygame.quit()

# # import pygame
# # import numpy as np

# # # ── Visual Constants ────────────────────────────────────────────────────────
# # SCREEN_WIDTH  = 1000
# # SCREEN_HEIGHT = 600
# # ROAD_WIDTH    = 100  
# # CAR_WIDTH     = 20
# # CAR_HEIGHT    = 12
# # FPS           = 1   # Slightly faster for smoother motion

# # # Colors
# # COLOR_ROAD    = (40, 40, 40)
# # COLOR_GRASS   = (30, 120, 30)
# # COLOR_MARKING = (200, 200, 200) 
# # COLOR_TEXT    = (255, 255, 255)

# # # Car Colors by Direction
# # COLORS = {
# #     "N": (255, 80, 80),   # Red-ish
# #     "S": (80, 255, 80),   # Green-ish
# #     "E": (80, 80, 255),   # Blue-ish
# #     "W": (255, 255, 80)   # Yellow-ish
# # }

# # class TrafficRenderer:
# #     def __init__(self):
# #         pygame.init()
# #         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# #         pygame.display.set_caption("Traffic RL: Two-Intersection Sync")
# #         self.clock = pygame.time.Clock()
# #         self.font = pygame.font.SysFont("Consolas", 20, bold=True)

# #     def _draw_car(self, x, y, direction):
# #         """Draws a car with headlights to show direction."""
# #         color = COLORS.get(direction, (200, 200, 200))
# #         w, h = (CAR_WIDTH, CAR_HEIGHT) if direction in ["E", "W"] else (CAR_HEIGHT, CAR_WIDTH)
            
# #         rect = pygame.Rect(x - w//2, y - h//2, w, h)
# #         pygame.draw.rect(self.screen, color, rect, border_radius=3)
# #         pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=3) 

# #         # Headlights
# #         headlight_color = (255, 255, 200)
# #         offsets = {
# #             "E": [(w//2, -h//4), (w//2, h//4)],
# #             "W": [(-w//2, -h//4), (-w//2, h//4)],
# #             "S": [(-h//4, w//2), (h//4, w//2)],
# #             "N": [(-h//4, -w//2), (h//4, -w//2)]
# #         }
# #         for dx, dy in offsets[direction]:
# #             pygame.draw.circle(self.screen, headlight_color, (int(x + dx), int(y + dy)), 2)

# #     def draw(self, env):
# #         self.screen.fill(COLOR_GRASS)
# #         il_x = SCREEN_WIDTH // 4
# #         ir_x = (SCREEN_WIDTH // 4) * 3
# #         mid_y = SCREEN_HEIGHT // 2
# #         lane_offset = ROAD_WIDTH // 4

# #         # 1. Draw Continuous Roads
# #         pygame.draw.rect(self.screen, COLOR_ROAD, (0, mid_y - ROAD_WIDTH//2, SCREEN_WIDTH, ROAD_WIDTH))
# #         pygame.draw.line(self.screen, COLOR_MARKING, (0, mid_y), (SCREEN_WIDTH, mid_y), 2)

# #         for x in [il_x, ir_x]:
# #             pygame.draw.rect(self.screen, COLOR_ROAD, (x - ROAD_WIDTH//2, 0, ROAD_WIDTH, SCREEN_HEIGHT))
# #             pygame.draw.line(self.screen, COLOR_MARKING, (x, 0), (x, SCREEN_HEIGHT), 2)

# #         # 2. Draw Queued & "Approaching" Cars
# #         # We calculate positions starting from the edge of the screen to avoid pops
# #         for i, x_center in enumerate([il_x, ir_x]):
# #             q = env.queues[i]
            
# #             # North Arm (S-bound) - Entering from top edge
# #             for k in range(int(q[0])):
# #                 self._draw_car(x_center + lane_offset, (mid_y - ROAD_WIDTH//2 - 20) - (k*25), "S")
            
# #             # South Arm (N-bound) - Entering from bottom edge
# #             for k in range(int(q[1])):
# #                 self._draw_car(x_center - lane_offset, (mid_y + ROAD_WIDTH//2 + 20) + (k*25), "N")

# #             # East Arm (W-bound) - IR enters from right edge; IL gets from corridor
# #             if i == 1: # Intersection Right
# #                 for k in range(int(q[2])):
# #                     self._draw_car(SCREEN_WIDTH - 20 - (k*25), mid_y - lane_offset, "W")
# #             else: # Intersection Left
# #                 for k in range(int(q[2])):
# #                     self._draw_car(x_center + ROAD_WIDTH//2 + 20 + (k*25), mid_y - lane_offset, "W")

# #             # West Arm (E-bound) - IL enters from left edge; IR gets from corridor
# #             if i == 0: # Intersection Left
# #                 for k in range(int(q[3])):
# #                     self._draw_car(20 + (k*25), mid_y + lane_offset, "E")
# #             else: # Intersection Right
# #                 for k in range(int(q[3])):
# #                     self._draw_car(x_center - ROAD_WIDTH//2 - 20 - (k*25), mid_y + lane_offset, "E")

# #             # 3. Traffic Lights
# #             p = env.phases[i]
# #             ns_c = (0, 255, 0) if p == 0 else (255, 0, 0)
# #             ew_c = (0, 255, 0) if p == 1 else (255, 0, 0)
# #             pygame.draw.circle(self.screen, ns_c, (x_center, mid_y - ROAD_WIDTH//2 - 15), 10)
# #             pygame.draw.circle(self.screen, ew_c, (x_center - ROAD_WIDTH//2 - 15, mid_y), 10)

# #         # 4. Draw Corridor (The Sync Area)
# #         corr_len = ir_x - il_x - ROAD_WIDTH
# #         step_w = corr_len / len(env.corridor_0to1)
# #         for idx, count in enumerate(env.corridor_0to1):
# #             if count > 0:
# #                 cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
# #                 self._draw_car(cx, mid_y + lane_offset, "E")
# #         for idx, count in enumerate(reversed(list(env.corridor_1to0))):
# #             if count > 0:
# #                 cx = il_x + ROAD_WIDTH//2 + (idx * step_w) + step_w//2
# #                 self._draw_car(cx, mid_y - lane_offset, "W")

# #         # 5. UI Stats
# #         info = self.font.render(f"STEP: {env.step_count} | TOTAL QUEUED: {int(env.queues.sum())}", True, COLOR_TEXT)
# #         self.screen.blit(info, (20, 20))

# #         pygame.display.flip()
# #         self.clock.tick(FPS)

# #     def close(self):
# #         pygame.quit()

# import pygame
# import math

# # ── Visual Constants ────────────────────────────────────────────────────────
# SCREEN_WIDTH  = 1200
# SCREEN_HEIGHT = 700
# ROAD_WIDTH    = 120
# CAR_W         = 28
# CAR_H         = 14
# FPS           = 1   # High FPS, env steps controlled separately

# # ── Color Palette ────────────────────────────────────────────────────────────
# C_BG          = (15,  20,  15)    # near-black green
# C_GRASS       = (28,  68,  28)
# C_ROAD        = (45,  45,  48)
# C_ROAD_EDGE   = (60,  60,  65)
# C_LANE_MARK   = (180, 180, 100)
# C_WHITE       = (240, 240, 240)
# C_PANEL       = (20,  25,  20)
# C_PANEL_LINE  = (50,  80,  50)

# CAR_COLORS = {
#     "N": (255,  90,  80),   # coral red
#     "S": ( 80, 220, 120),   # mint green
#     "E": ( 80, 160, 255),   # sky blue
#     "W": (255, 210,  60),   # amber
# }
# HEADLIGHT_COLOR = (255, 255, 210)

# # ── Layout (computed once) ───────────────────────────────────────────────────
# IL_X   = SCREEN_WIDTH  // 4        # left  intersection center x
# IR_X   = (SCREEN_WIDTH * 3) // 4  # right intersection center x
# MID_Y  = SCREEN_HEIGHT // 2        # road center y
# RH     = ROAD_WIDTH // 2
# LO     = ROAD_WIDTH // 4           # lane offset from center

# # Queue car spacing (px between car fronts)
# CAR_SPACING = 32


# def _lerp_color(a, b, t):
#     return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# class TrafficRenderer:
#     def __init__(self):
#         pygame.init()
#         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#         pygame.display.set_caption("Traffic RL — Two-Intersection")
#         self.clock  = pygame.time.Clock()
#         self.font_lg = pygame.font.SysFont("Consolas", 22, bold=True)
#         self.font_sm = pygame.font.SysFont("Consolas", 15)
#         self.font_xs = pygame.font.SysFont("Consolas", 12)
#         self._step  = 0

#     # ── Drawing Primitives ───────────────────────────────────────────────────

#     def _car(self, x, y, direction, alpha=255):
#         """Draw a single car rectangle with headlights."""
#         color = CAR_COLORS.get(direction, C_WHITE)
#         horiz = direction in ("E", "W")
#         w = CAR_W if horiz else CAR_H
#         h = CAR_H if horiz else CAR_W

#         # Body
#         rect = pygame.Rect(int(x - w / 2), int(y - h / 2), w, h)
#         surf = pygame.Surface((w, h), pygame.SRCALPHA)
#         pygame.draw.rect(surf, (*color, alpha), (0, 0, w, h), border_radius=3)
#         # Outline
#         pygame.draw.rect(surf, (0, 0, 0, alpha), (0, 0, w, h), 1, border_radius=3)
#         self.screen.blit(surf, rect.topleft)

#         # Headlights
#         offsets = {
#             "E": [(w // 2 - 2, -h // 4), (w // 2 - 2,  h // 4)],
#             "W": [(-w // 2 + 2, -h // 4), (-w // 2 + 2,  h // 4)],
#             "S": [(-h // 4,  w // 2 - 2), ( h // 4,  w // 2 - 2)],
#             "N": [(-h // 4, -w // 2 + 2), ( h // 4, -w // 2 + 2)],
#         }
#         for dx, dy in offsets[direction]:
#             pygame.draw.circle(self.screen, HEADLIGHT_COLOR, (int(x + dx), int(y + dy)), 2)

#     def _traffic_light(self, cx, cy, phase):
#         """Draw a compact traffic-light indicator."""
#         # NS light (top of vertical road arm)
#         ns_green = (phase == 0)
#         ew_green = (phase == 1)

#         # NS indicator — above intersection, on the vertical road
#         ns_pos = (cx, cy - RH - 22)
#         # EW indicator — left of intersection, on the horizontal road
#         ew_pos = (cx - RH - 22, cy)

#         for pos, is_green in [(ns_pos, ns_green), (ew_pos, ew_green)]:
#             # Housing
#             pygame.draw.rect(self.screen, (30, 30, 30),
#                              (pos[0] - 10, pos[1] - 10, 20, 20), border_radius=4)
#             col = (30, 220, 80) if is_green else (220, 50, 40)
#             pygame.draw.circle(self.screen, col, pos, 7)
#             # Glow ring
#             glow = _lerp_color(col, (0, 0, 0), 0.4)
#             pygame.draw.circle(self.screen, glow, pos, 9, 2)

#         # Small NS / EW label
#         lbl_ns = self.font_xs.render("NS", True, (200, 200, 200))
#         lbl_ew = self.font_xs.render("EW", True, (200, 200, 200))
#         self.screen.blit(lbl_ns, (ns_pos[0] + 12, ns_pos[1] - 7))
#         self.screen.blit(lbl_ew, (ew_pos[0] + 12, ew_pos[1] - 7))

#     # ── Road Drawing ─────────────────────────────────────────────────────────

#     def _draw_roads(self):
#         # Horizontal road
#         pygame.draw.rect(self.screen, C_ROAD_EDGE,
#                          (0, MID_Y - RH - 2, SCREEN_WIDTH, ROAD_WIDTH + 4))
#         pygame.draw.rect(self.screen, C_ROAD,
#                          (0, MID_Y - RH, SCREEN_WIDTH, ROAD_WIDTH))

#         # Vertical roads
#         for cx in (IL_X, IR_X):
#             pygame.draw.rect(self.screen, C_ROAD_EDGE,
#                              (cx - RH - 2, 0, ROAD_WIDTH + 4, SCREEN_HEIGHT))
#             pygame.draw.rect(self.screen, C_ROAD,
#                              (cx - RH, 0, ROAD_WIDTH, SCREEN_HEIGHT))

#         # Re-fill intersections on top (they get split by the vertical draw)
#         for cx in (IL_X, IR_X):
#             pygame.draw.rect(self.screen, C_ROAD,
#                              (cx - RH, MID_Y - RH, ROAD_WIDTH, ROAD_WIDTH))

#         # Center dashed lane markings
#         dash, gap = 18, 10
#         # Horizontal
#         for x in range(0, SCREEN_WIDTH, dash + gap):
#             if abs(x - IL_X) > RH and abs(x - IR_X) > RH:
#                 pygame.draw.rect(self.screen, C_LANE_MARK,
#                                  (x, MID_Y - 1, dash, 2))
#         # Vertical (left intersection)
#         for y in range(0, SCREEN_HEIGHT, dash + gap):
#             if abs(y - MID_Y) > RH:
#                 pygame.draw.rect(self.screen, C_LANE_MARK,
#                                  (IL_X - 1, y, 2, dash))
#                 pygame.draw.rect(self.screen, C_LANE_MARK,
#                                  (IR_X - 1, y, 2, dash))

#     # ── Queue Drawing ─────────────────────────────────────────────────────────

#     def _draw_queue(self, count, start_x, start_y, dx, dy, direction):
#         """
#         Draw `count` cars in a queue starting at (start_x, start_y),
#         moving in direction (dx, dy) per slot.
#         Cars are spaced CAR_SPACING apart and capped to avoid going off screen.
#         """
#         n = min(int(count), 12)   # never draw more than 12 (avoids off-screen mess)
#         for k in range(n):
#             cx = start_x + dx * k
#             cy = start_y + dy * k
#             # fade the furthest cars slightly
#             alpha = 255 if k < 6 else max(80, 255 - (k - 6) * 40)
#             self._car(cx, cy, direction, alpha)

#         # Count badge if more than drawn
#         total = int(count)
#         if total > n:
#             bx = int(start_x + dx * n)
#             by = int(start_y + dy * n)
#             badge = self.font_xs.render(f"+{total - n}", True, (255, 220, 80))
#             self.screen.blit(badge, (bx - 10, by - 8))

#     # ── Corridor Drawing ──────────────────────────────────────────────────────

#     def _draw_corridor(self, env):
#         """Draw cars travelling between the two intersections."""
#         corr_start_x = IL_X + RH + 10
#         corr_end_x   = IR_X - RH - 10
#         corr_len      = corr_end_x - corr_start_x
#         n_cells       = len(env.corridor_0to1)
#         cell_w        = corr_len / n_cells if n_cells > 0 else 1

#         # Eastbound (0→1): top lane of horizontal road (MID_Y - LO)
#         for idx, count in enumerate(env.corridor_0to1):
#             if count > 0:
#                 cx = corr_start_x + idx * cell_w + cell_w / 2
#                 cy = MID_Y - LO
#                 self._car(cx, cy, "E")
#                 if count > 1:
#                     lbl = self.font_xs.render(f"x{int(count)}", True, (255, 220, 80))
#                     self.screen.blit(lbl, (int(cx) - 8, int(cy) - 18))

#         # Westbound (1→0): bottom lane of horizontal road (MID_Y + LO)
#         cells_1to0 = list(reversed(list(env.corridor_1to0)))
#         for idx, count in enumerate(cells_1to0):
#             if count > 0:
#                 cx = corr_start_x + idx * cell_w + cell_w / 2
#                 cy = MID_Y + LO
#                 self._car(cx, cy, "W")
#                 if count > 1:
#                     lbl = self.font_xs.render(f"x{int(count)}", True, (255, 220, 80))
#                     self.screen.blit(lbl, (int(cx) - 8, int(cy) - 18))

#     # ── HUD / Stats Panel ─────────────────────────────────────────────────────

#     def _draw_hud(self, env):
#         panel_h = 80
#         panel   = pygame.Surface((SCREEN_WIDTH, panel_h), pygame.SRCALPHA)
#         panel.fill((10, 15, 10, 200))
#         self.screen.blit(panel, (0, 0))
#         pygame.draw.line(self.screen, C_PANEL_LINE, (0, panel_h), (SCREEN_WIDTH, panel_h), 1)

#         # Step & queue totals
#         step_txt  = self.font_lg.render(
#             f"STEP: {env.step_count}   |   TOTAL QUEUED: {int(env.queues.sum())}",
#             True, C_WHITE)
#         self.screen.blit(step_txt, (20, 12))

#         # Per-intersection queue breakdown
#         for i, (cx, label) in enumerate([(IL_X, "LEFT  INT"), (IR_X, "RIGHT INT")]):
#             q     = env.queues[i]
#             phase = env.phases[i]
#             dirs  = ["N↓", "S↑", "W←", "E→"]
#             cols  = [CAR_COLORS["S"], CAR_COLORS["N"], CAR_COLORS["W"], CAR_COLORS["E"]]
#             phase_str = "NS green" if phase == 0 else "EW green"

#             x_off = 20 + i * 580
#             hdr   = self.font_sm.render(f"{label}  [{phase_str}]", True, (180, 220, 180))
#             self.screen.blit(hdr, (x_off, 40))

#             for j, (d, c, v) in enumerate(zip(dirs, cols, q)):
#                 bar_w = int(min(v, 20) * 5)
#                 bar_x = x_off + j * 130
#                 pygame.draw.rect(self.screen, c,
#                                  (bar_x, 58, bar_w, 10), border_radius=3)
#                 txt = self.font_xs.render(f"{d}:{int(v)}", True, c)
#                 self.screen.blit(txt, (bar_x, 58))

#         pygame.draw.line(self.screen, C_PANEL_LINE,
#                          (0, panel_h), (SCREEN_WIDTH, panel_h), 1)

#     # ── Main Draw ─────────────────────────────────────────────────────────────

#     def draw(self, env):
#         # Background
#         self.screen.fill(C_GRASS)

#         # Roads
#         self._draw_roads()

#         # Per-intersection queues & lights
#         for i, cx in enumerate([IL_X, IR_X]):
#             q = env.queues[i]

#             # --- North arm: S-bound cars come FROM the north, stop at intersection ---
#             # They stack upward from the intersection edge
#             self._draw_queue(
#                 q[0],
#                 cx + LO,                  # right lane of vertical road
#                 MID_Y - RH - CAR_W // 2, # just above intersection
#                 0, -CAR_SPACING,          # stack upward
#                 "S"
#             )

#             # --- South arm: N-bound cars come FROM the south ---
#             self._draw_queue(
#                 q[1],
#                 cx - LO,
#                 MID_Y + RH + CAR_W // 2, # just below intersection
#                 0, CAR_SPACING,           # stack downward
#                 "N"
#             )

#             # --- East arm: W-bound cars ---
#             # Left intersection: cars come from the corridor side (right of IL)
#             # Right intersection: cars come from the right edge of screen
#             if i == 0:
#                 start_x = cx + RH + CAR_H // 2
#                 dx      = CAR_SPACING
#             else:
#                 start_x = SCREEN_WIDTH - CAR_H // 2 - 10
#                 dx      = -CAR_SPACING  # NO! these are W-bound, stack toward left
#                 start_x = IR_X + RH + CAR_H // 2
#                 dx      = CAR_SPACING

#             self._draw_queue(
#                 q[2],
#                 start_x,
#                 MID_Y - LO,
#                 dx, 0,
#                 "W"
#             )

#             # --- West arm: E-bound cars ---
#             if i == 0:
#                 start_x = IL_X - RH - CAR_H // 2
#                 dx      = -CAR_SPACING
#             else:
#                 start_x = IR_X - RH - CAR_H // 2
#                 dx      = -CAR_SPACING

#             self._draw_queue(
#                 q[3],
#                 start_x,
#                 MID_Y + LO,
#                 dx, 0,
#                 "E"
#             )

#             # Traffic lights
#             self._traffic_light(cx, MID_Y, env.phases[i])

#         # Corridor cars (between intersections)
#         self._draw_corridor(env)

#         # HUD on top
#         self._draw_hud(env)

#         pygame.display.flip()
#         self.clock.tick(FPS)

#     def close(self):
#         pygame.quit()
import pygame
import math

# ── Visual Constants ────────────────────────────────────────────────────────
SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 700
ROAD_WIDTH    = 120
CAR_W         = 28
CAR_H         = 14
FPS           = 1

# ── Color Palette ────────────────────────────────────────────────────────────
C_BG          = (15,  20,  15)
C_GRASS       = (28,  68,  28)
C_ROAD        = (45,  45,  48)
C_LANE_MARK   = (180, 180, 100)
C_WHITE       = (240, 240, 240)

CAR_COLORS = {
    "N": (255,  90,  80),
    "S": ( 80, 220, 120),
    "E": ( 80, 160, 255),
    "W": (255, 210,  60),
}

# ── Layout ───────────────────────────────────────────────────────────────────
IL_X   = SCREEN_WIDTH  // 4
IR_X   = (SCREEN_WIDTH * 3) // 4
MID_Y  = SCREEN_HEIGHT // 2
RH     = ROAD_WIDTH // 2
LO     = ROAD_WIDTH // 4
CAR_SPACING = 32

class TrafficRenderer:
    def __init__(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Traffic RL — India LHT Final")
        self.clock   = pygame.time.Clock()
        self.font_lg = pygame.font.SysFont("Consolas", 22, bold=True)

    def _car(self, x, y, direction):
        color = CAR_COLORS.get(direction, C_WHITE)
        horiz = direction in ("E", "W")
        w, h = (CAR_W, CAR_H) if horiz else (CAR_H, CAR_W)
        rect = pygame.Rect(int(x - w / 2), int(y - h / 2), w, h)
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (*color, 255), (0, 0, w, h), border_radius=3)
        pygame.draw.rect(surf, (0, 0, 0, 255), (0, 0, w, h), 1, border_radius=3)
        self.screen.blit(surf, rect.topleft)

    def _traffic_light(self, cx, cy, phase):
        ns_green, ew_green = (phase == 0), (phase == 1)
        # Lights at stop lines
        ns_pos = (cx - LO - 15, cy + RH + 15)
        ew_pos = (cx - RH - 15, cy - LO - 15)
        for pos, is_green in [(ns_pos, ns_green), (ew_pos, ew_green)]:
            col = (30, 220, 80) if is_green else (220, 50, 40)
            pygame.draw.circle(self.screen, (30, 30, 30), pos, 10)
            pygame.draw.circle(self.screen, col, pos, 7)

    def _draw_roads(self):
        self.screen.fill(C_GRASS)
        pygame.draw.rect(self.screen, C_ROAD, (0, MID_Y - RH, SCREEN_WIDTH, ROAD_WIDTH))
        for cx in (IL_X, IR_X):
            pygame.draw.rect(self.screen, C_ROAD, (cx - RH, 0, ROAD_WIDTH, SCREEN_HEIGHT))
        
        dash, gap = 18, 10
        for x in range(0, SCREEN_WIDTH, dash + gap):
            pygame.draw.rect(self.screen, C_LANE_MARK, (x, MID_Y - 1, dash, 2))
        for cx in (IL_X, IR_X):
            for y in range(0, SCREEN_HEIGHT, dash + gap):
                pygame.draw.rect(self.screen, C_LANE_MARK, (cx - 1, y, 2, dash))

    def _draw_queue(self, count, stop_x, stop_y, dx, dy, direction):
        """Cars fill up from the stop-line (stop_x, stop_y) backwards away from intersection."""
        n = min(int(count), 20)
        for k in range(n):
            # dx/dy should point AWAY from the intersection to stack them backwards
            self._car(stop_x + dx * k, stop_y + dy * k, direction)

    def _draw_corridor(self, env):
        corr_start_x = IL_X + RH + 15
        corr_end_x   = IR_X - RH - 15
        n_cells = len(env.corridor_0to1)
        cell_w = (corr_end_x - corr_start_x) / n_cells if n_cells > 0 else 1

        # Eastbound (→): TOP lane 
        for idx, count in enumerate(env.corridor_0to1):
            if count > 0:
                cx = corr_start_x + idx * cell_w + (cell_w / 2)
                self._car(cx, MID_Y - LO, "E")

        # Westbound (←): BOTTOM lane
        cells_1to0 = list(reversed(list(env.corridor_1to0)))
        for idx, count in enumerate(cells_1to0):
            if count > 0:
                cx = corr_start_x + idx * cell_w + (cell_w / 2)
                self._car(cx, MID_Y + LO, "W")

    def draw(self, env):
        self._draw_roads()

        for i, cx in enumerate([IL_X, IR_X]):
            q = env.queues[i]

            # NORTH (↑): Stop-line at bottom of intersection, stack downwards to screen edge
            self._draw_queue(q[0], cx - LO, MID_Y + RH + 20, 0, CAR_SPACING, "N")

            # SOUTH (↓): Stop-line at top of intersection, stack upwards to screen edge
            self._draw_queue(q[1], cx + LO, MID_Y - RH - 20, 0, -CAR_SPACING, "S")

            # EAST (→): Stop-line at left of intersection, stack leftwards to screen edge
            if i == 0:
                self._draw_queue(q[2], cx - RH - 20, MID_Y - LO, -CAR_SPACING, 0, "E")

            # WEST (←): Stop-line at right of intersection, stack rightwards to screen edge
            if i == 1:
                self._draw_queue(q[3], cx + RH + 20, MID_Y + LO, CAR_SPACING, 0, "W")

            self._traffic_light(cx, MID_Y, env.phases[i])

        self._draw_corridor(env)

        step_txt = self.font_lg.render(f"STEP: {env.step_count} | QUEUED: {int(env.queues.sum())}", True, C_WHITE)
        self.screen.blit(step_txt, (20, 20))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()