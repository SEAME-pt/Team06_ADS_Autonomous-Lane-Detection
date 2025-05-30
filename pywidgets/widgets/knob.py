from pywidgets.core.widget import Widget
from pywidgets.core.skin import *
import pygame
import math

class Knob(Widget):
    def __init__(self, x, y, radius=30, min_value=0.0, max_value=1.0, value=0.0, on_change=None):
        super().__init__(x, y, radius * 2, radius * 2)
        self.radius = radius
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.on_change = on_change
        self.dragging = False
        self.start_y = 0
        self.start_value = 0.0

    def render(self, surface):
        cx = self.get_x() + self.radius
        cy = self.get_y() + self.radius
        outline = self.skin.get_property(SkinProps.BORDER_COLOR, (180, 180, 180))
        fill = self.skin.get_property(SkinProps.BACKGROUND_COLOR, (30, 30, 30))
        mark = self.skin.get_property(SkinProps.TEXT_COLOR, (255, 255, 255))

        pygame.draw.circle(surface, fill, (cx, cy), self.radius)
        pygame.draw.circle(surface, outline, (cx, cy), self.radius, 2)

        # ponteiro
        pct = (self.value - self.min_value) / (self.max_value - self.min_value)
        angle = math.radians(135) + pct * math.radians(270)
        end_x = cx + math.cos(angle) * (self.radius - 6)
        end_y = cy + math.sin(angle) * (self.radius - 6)
        pygame.draw.line(surface, mark, (cx, cy), (end_x, end_y), 3)

    def on_mouse_down(self, x, y):
        if self.bound.collidepoint(x, y):
            self.dragging = True
            self.start_y = y
            self.start_value = self.value
            return True
        return False

    def on_mouse_up(self, x, y):
        self.dragging = False
        return False

    def on_mouse_move(self, x, y):
        if self.dragging:
            dy = self.start_y - y
            delta = dy * 0.005 * (self.max_value - self.min_value)
            new_value = max(self.min_value, min(self.start_value + delta, self.max_value))
            if new_value != self.value:
                self.value = new_value
                if self.on_change:
                    self.on_change(self.value)
            return True
        return False

    def set_value(self, value):
        self.value = max(self.min_value, min(self.max_value, value))

    def get_value(self):
        return self.value