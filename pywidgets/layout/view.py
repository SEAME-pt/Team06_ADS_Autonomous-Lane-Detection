import pygame
from pywidgets.core.layout import Layout
from pywidgets.core.skin import *
from pywidgets.widgets.scroll import  Scrollbar

class ViewLayout(Layout):
    def __init__(self, x, y, width, height, content_width=None, content_height=None):
        super().__init__(x, y, width, height)
        self.content_width = content_width if content_width else width
        self.content_height = content_height if content_height else height
        self.scroll_x = 0
        self.scroll_y = 0
        self.scroll_speed = 20

        self.scrollbar_v = Scrollbar(0, 0, height, vertical=True,
                                     content_size=self.content_height, view_size=height,
                                     scroll_pos=0, on_scroll=self._on_scroll_y)
        #self.scrollbar_v.parent = self #vamos actuzliar no updatte ,ate ver 

        self.scrollbar_h = Scrollbar(0, 0, width, vertical=False,
                                     content_size=self.content_width, view_size=width,
                                     scroll_pos=0, on_scroll=self._on_scroll_x)
        #self.scrollbar_h.parent = self

    def _on_scroll_y(self, value):
        self.scroll_y = int(value)

    def _on_scroll_x(self, value):
        self.scroll_x = int(value)

    def update(self, dt):
        self.bound.x = self.get_x()
        self.bound.y = self.get_y()
        self.bound.width = self.rect.width
        self.bound.height = self.rect.height


        self.scrollbar_v.rect.x = self.get_x() + self.rect.width - self.scrollbar_v.rect.width
        self.scrollbar_v.rect.y = self.get_y()
        self.scrollbar_v.rect.height = self.rect.height

        self.scrollbar_h.rect.x = self.get_x()
        self.scrollbar_h.rect.y = self.get_y() + self.rect.height - self.scrollbar_h.rect.height
        self.scrollbar_h.rect.width = self.rect.width

        self.scrollbar_v.content_size = self.content_height
        self.scrollbar_v.view_size = self.rect.height
        self.scrollbar_v.update(dt)

        self.scrollbar_h.content_size = self.content_width
        self.scrollbar_h.view_size = self.rect.width
        self.scrollbar_h.update(dt)

        for widget in self.widgets:
            if widget.enabled:
                widget.update(dt)

    def render(self, surface):
        clip = surface.get_clip()
        surface.set_clip(self.bound)
        self.scrollbar_v.skin = self.skin
        self.scrollbar_h.skin = self.skin

        for widget in self.widgets:
            if widget.visible:
                ox, oy = widget.rect.x, widget.rect.y
                widget.rect.x -= self.scroll_x
                widget.rect.y -= self.scroll_y
                widget.render(surface)
                widget.rect.x, widget.rect.y = ox, oy

        surface.set_clip(clip)

        if self.content_height > self.rect.height:
            self.scrollbar_v.render(surface)
        if self.content_width > self.rect.width:
            self.scrollbar_h.render(surface)
        pygame.draw.rect(surface, (255,0,0), self.bound, 1)

    def on_mouse_down(self, x, y):
        if self.content_height > self.rect.height and self.scrollbar_v.on_mouse_down(x, y): return True
        if self.content_width > self.rect.width and self.scrollbar_h.on_mouse_down(x, y): return True

        if not self.bound.collidepoint(x, y):
            return False
        for widget in reversed(self.widgets):
            if widget.visible and widget.enabled:
                if widget.on_mouse_down(x + self.scroll_x, y + self.scroll_y):
                    return True
        return False

    def on_mouse_up(self, x, y):
        self.scrollbar_v.on_mouse_up(x, y)
        self.scrollbar_h.on_mouse_up(x, y)
        for widget in reversed(self.widgets):
            if widget.visible and widget.enabled:
                widget.on_mouse_up(x + self.scroll_x, y + self.scroll_y)

    def on_mouse_move(self, x, y):
        if self.scrollbar_v.on_mouse_move(x, y): return True
        if self.scrollbar_h.on_mouse_move(x, y): return True
        for widget in reversed(self.widgets):
            if widget.visible and widget.enabled:
                if widget.on_mouse_move(x + self.scroll_x, y + self.scroll_y):
                    return True
        return False

    def on_mouse_wheel(self, delta):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if self.scrollbar_v.bound.collidepoint(mouse_x, mouse_y):
            return self.scrollbar_v.on_mouse_wheel(delta)
        if self.scrollbar_h.bound.collidepoint(mouse_x, mouse_y):
            return self.scrollbar_h.on_mouse_wheel(delta)
        if not self.bound.collidepoint(mouse_x, mouse_y):
            return False
        self.scroll_y = max(0, min(self.scroll_y - delta * self.scroll_speed, max(0, self.content_height - self.rect.height)))
        self.scrollbar_v.scroll_pos = self.scroll_y
        return True

    def set_scroll(self, x, y):
        self.scroll_x = max(0, min(x, self.content_width - self.rect.width))
        self.scroll_y = max(0, min(y, self.content_height - self.rect.height))
        self.scrollbar_v.scroll_pos = self.scroll_y
        self.scrollbar_h.scroll_pos = self.scroll_x

    def on_resize(self, w, h):
        super().on_resize(w, h)
        self.set_size(w, h)