import pygame
from pywidgets.core.layout import Layout
from pywidgets.core.skin import *


class VerticalBox(Layout):
    def __init__(self, x, y, width, spacing=8):
        super().__init__(x, y, width, 0)
 
        self.spacing = spacing

    def add(self, widget):
        widget.parent = self
        widget.skin = self.skin
        self.widgets.append(widget)
        self._update_layout()
        return widget

    def _update_layout(self):
        y_offset = 0
        for w in self.widgets:
            w.set_position(0, y_offset)
            w.rect.width = self.rect.width
            y_offset += w.rect.height + self.spacing
        self.rect.height = y_offset - self.spacing if self.widgets else 0

    def update(self, dt):
        super().update(dt)
        self._update_layout()
        for w in self.widgets:
            w.update(dt)

    def render(self, surface):
        for w in self.widgets:
            w.skin = self.skin
            w.render(surface)

    def on_mouse_down(self, x, y):
        for w in reversed(self.widgets):
            if w.on_mouse_down(x, y):
                return True
        return False

    def on_mouse_move(self, x, y):
        for w in reversed(self.widgets):
            if w.on_mouse_move(x, y):
                return True
        return False

    def on_mouse_up(self, x, y):
        for w in self.widgets:
            w.on_mouse_up(x, y)
        return False



class VerticalBoxAuto(Layout):
    def __init__(self, x, y, width, height, spacing=4, margins=None):
        super().__init__(x, y, width, height)
        self.spacing = spacing
        self.entries = []
        self.margins = margins if margins else Margins()
 

    def set_margins(self, margins):
        self.margins = margins
        self._update_layout()

    def add(self, widget, percent=1.0):
        self.entries.append((widget, percent))
        widget.parent = self
        widget.skin = self.skin
        self.widgets.append(widget)
        self._update_layout()
        return widget

    def _update_layout(self):
        total_spacing = self.spacing * (len(self.entries) - 1)
        usable_height =  (self.rect.height-self.rect.y) - self.margins.top - self.margins.bottom - total_spacing- self.margins.bottom
        
        y_offset =  self.margins.top

        for widget, percent in self.entries:
            widget_height = int(usable_height * percent)
            widget.set_position(self.margins.left, y_offset)
            widget.set_size(self.rect.width - self.margins.left - self.margins.right, widget_height)
            y_offset += widget_height + self.spacing

    def on_resize(self, w, h):
        self.rect.height = h
        self._update_layout()