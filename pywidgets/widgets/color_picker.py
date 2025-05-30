from pywidgets.core.widget import Widget
from pywidgets.core.skin import *
import pygame
import colorsys

class ColorMapSurface(Widget):
    def __init__(self, x, y, size=128, hue=0.0, on_change=None):
        super().__init__(x, y, size, size)
        self.hue = hue
        self.s = 1.0
        self.v = 1.0
        self.surface = pygame.Surface((size, size))
        self.on_change = on_change
        self.pressed = False
        self._regen_surface()

    def _regen_surface(self):
        for x in range(self.rect.width):
            for y in range(self.rect.height):
                s = x / self.rect.width
                v = 1 - y / self.rect.height
                r, g, b = colorsys.hsv_to_rgb(self.hue, s, v)
                self.surface.set_at((x, y), (int(r * 255), int(g * 255), int(b * 255)))

    def set_hue(self, hue):
        self.hue = hue
        self._regen_surface()

    def get_color(self):
        r, g, b = colorsys.hsv_to_rgb(self.hue, self.s, self.v)
        return int(r * 255), int(g * 255), int(b * 255)

    def get_hsv(self):
        return self.hue, self.s, self.v
 

    def render(self, surface):
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.height = self.rect.height
        self.bound.width = self.rect.width
        surface.blit(self.surface, (self.get_x(), self.get_y()))
        # marcador
        cx = int(self.get_x() + self.s * self.rect.width)
        cy = int(self.get_y() + (1 - self.v) * self.rect.height)
        pygame.draw.circle(surface, (0, 0, 0), (cx, cy), 5, 1)
        #pygame.draw.rect(surface, (255,255,255), self.bound, width=1)

    def on_mouse_down(self, x, y):
        if  self.bound.collidepoint(x, y):
            self._update_color(x, y)
            self.pressed = True
            return True
        return False
    
    def on_mouse_up(self, x, y):
        self.pressed = False
        return False

    def on_mouse_move(self, x, y):
        if self.bound.collidepoint(x, y) and self.pressed:
            self._update_color(x, y)
            return True
        self.pressed = False
        return False

    def _update_color(self, x, y):
        rx = min(max(x - self.get_x(), 0), self.rect.width - 1)
        ry = min(max(y - self.get_y(), 0), self.rect.height - 1)
        self.s = rx / self.rect.width
        self.v = 1 - (ry / self.rect.height)
        if self.on_change:
            self.on_change(self.get_color())
        return True


class HueSlider(Widget):
    def __init__(self, x, y, height=128, on_change=None):
        super().__init__(x, y, 20, height)
        self.hue = 0.0
        self.on_change = on_change
        self.pressed = False

    def render(self, surface):
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.height = self.rect.height
        self.bound.width = self.rect.width
        for y in range(self.rect.height):
            hue = y / self.rect.height
            r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
            pygame.draw.line(surface, (int(r * 255), int(g * 255), int(b * 255)),
                             (self.get_x(), self.get_y() + y),
                             (self.get_x() + self.rect.width, self.get_y() + y))

        # marcador
        hy = self.get_y() + int(self.hue * self.rect.height)
        pygame.draw.rect(surface, (0, 0, 0), (self.get_x(), hy - 1, self.rect.width, 2))
        #pygame.draw.rect(surface, (0,255,255), self.bound, width=1)

    def on_mouse_down(self, x, y):
        if self.bound.collidepoint(x, y):
            self.pressed = True
            self._update_hue(y)
            return True
        return False

    def on_mouse_move(self, x, y):
        if self.bound.collidepoint(x, y) and self.pressed:
            self._update_hue(y)
            return True
        return False

    def on_mouse_up(self, x, y):
        self.pressed = False
        return False

    def _update_hue(self, y):
        ry = min(max(y - self.get_y(), 0), self.rect.height - 1)
        self.hue = ry / self.rect.height
        if self.on_change:
            self.on_change(self.hue)
        


class ColorPicker(Widget):
    def __init__(self, x, y, size=128, on_change=None):
        super().__init__(x, y, size + 30, size + 40)
        self.map = ColorMapSurface(0, 0, size, on_change=self._color_updated)
        self.map.parent = self
        self.slider = HueSlider(size + 2, 0, size-1, on_change=self._hue_changed)
        self.slider.parent = self
        self.preview = pygame.Rect(self.get_x(), self.get_y() + size +1, size + 24, 20)
        self._color = (255, 0, 0)
        self.on_change = on_change
        self.map.parent = self
        self.slider.parent = self
        self.size = size
        self.pressed = False

    def update(self, dt):
        self.map.update(dt)
        self.slider.update(dt)
        self.bound.width = self.rect.width   - 30
        self.bound.height = self.rect.height - 40

    def render(self, surface):
        self.map.skin = self.skin
        self.slider.skin = self.skin
        self.map.render(surface)
        self.slider.render(surface)
        self.preview = pygame.Rect(self.get_x(), self.get_y() + self.size +1, self.size + 24, 20)
        pygame.draw.rect(surface, self._color, self.preview)
        #pygame.draw.rect(surface, (0, 0, 0), self.preview, 1)
        #pygame.draw.rect(surface, (255,0,0), self.rect, width=1)
        #pygame.draw.rect(surface, (255,255,255), self.bound, width=1)

    def on_mouse_down(self, x, y):
        self.pressed = self.bound.collidepoint(x, y)      
        if self.map.on_mouse_down(x, y):
            return True
        if self.slider.on_mouse_down(x, y):
            return True
        return False

    def on_mouse_move(self, x, y):
        return self.map.on_mouse_move(x, y) or self.slider.on_mouse_move(x, y)
    
    def on_mouse_up(self, x, y):
        self.map.on_mouse_up(x, y)
        self.slider.on_mouse_up(x, y)
        self.pressed = False
        return False

    def _hue_changed(self, hue):
        self.map.set_hue(hue)
        self._color = self.map.get_color()
        if self.on_change:
            self.on_change(self._color)

    def _color_updated(self, color):
        self._color = color
        if self.on_change:
            self.on_change(color)

    def get_color(self):
        return self._color